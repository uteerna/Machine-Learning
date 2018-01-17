import math
import numpy


# This method normalizes the weights by dividing each element of the numpy array by the total.
def normalizeWeights(weights):
    total = numpy.sum(weights)
    return weights / total


# Multiply weights of correctly classified elements by error/(1-error) and then normalize all weights
def updateWeights(weights, max_D, error):
    mul = float(error) / float(1 - error)
    D = numpy.array(max_D, dtype=object)
    D[D == True] = mul
    D[D == False] = 1
    D = D.astype(float)
    new_weights = numpy.multiply(weights, D)
    new_weights = normalizeWeights(new_weights)
    return new_weights


# This method will return a list of 'stump_count' number of decision stumps
# Each decision stump is a pair of columns from the array of 192 values
# It also returns a list of corresponding errors for each stump which is useful when calculating the weight of the stump.
def getStumpsandErrors(train_data_array, array_x, stump_count):
    weights = numpy.array([float(1) / float(len(array_x)) for x in range(len(array_x))])
    stumps_list = []
    errors_list = []
    for s in range(stump_count):
        max_0 = -1
        max_i = -1
        max_j = -1
        max_D = -1
        error = 0
        for i in range(len(train_data_array[0])):
            for j in range(len(train_data_array[0])):
                if i != j and (i, j) not in stumps_list and (j, i) not in stumps_list:
                    C = numpy.greater(train_data_array[:, i], train_data_array[:, j])
                    D = (C == array_x)
                    D = D.astype(int)
                    total = numpy.sum(numpy.multiply(weights, D))
                    if max_0 < total:
                        max_0 = total
                        max_i = i
                        max_j = j
                        max_D = D.astype(bool)
                        error = (float(len(array_x)) - float(numpy.sum(C == array_x))) / float(len(array_x))
        weights = updateWeights(weights, max_D, error)
        stumps_list.append((max_i, max_j))
        errors_list.append(error)
    print stumps_list
    print errors_list
    return (stumps_list, errors_list)


# This method returns the value which we get be doing a summation of all the stump*weight,
# where the stump has a value of -1 or 1 based on the condition.
# If it is positive(>0) the resultant vote is classified as yes for that orientation.
# Else it is classified as 'no'
def getVote(stumps_x, error_x_weight_list, row, stump_count):
    votes = []
    for i in range(stump_count):
        if row[stumps_x[i][0]] > row[stumps_x[i][1]]:
            votes.append(+1)
        else:
            votes.append(-1)
    final_vote = numpy.multiply(numpy.array(votes), numpy.array(error_x_weight_list[:stump_count]))
    return numpy.sum(final_vote)


# This method gets the set of decision stumps and weights based on input file and stump count.
def trainHypothesis(train_file, stump_count):
    # This method appends the given boolean values to the lists
    def appendBoolean(a, b, c, d):
        list_0.append(a)
        list_90.append(b)
        list_180.append(c)
        list_270.append(d)

    print "Starting to read training data"

    # A list of list of the the training data having 192 columns in each row.
    # The number of rows is equal to the number of lines in the training data file
    train_data_list = []

    # These are 4 lists of booleans where index represents the line number in the training set
    # The value represents whether the image at the index is in that orientation
    # Thus for any given index only one of the four lists will have 'True'
    # All the other lists will have 'False' at that index
    list_0 = []
    list_90 = []
    list_180 = []
    list_270 = []

    with open(train_file) as f:
        content = f.readlines()

    # Here we read the training data line by line and populate the training data lists
    for line in content:
        if line.split()[1] == '0':
            appendBoolean(True, False, False, False)
        elif line.split()[1] == '90':
            appendBoolean(False, True, False, False)
        elif line.split()[1] == '180':
            appendBoolean(False, False, True, False)
        elif line.split()[1] == '270':
            appendBoolean(False, False, False, True)
        train_data_list.append([int(x) for x in line.split()[2:]])

    # Creating 4 numpy arrays from the list of booleans
    array_0 = numpy.array(list_0)
    array_90 = numpy.array(list_90)
    array_180 = numpy.array(list_180)
    array_270 = numpy.array(list_270)

    # A 2D numpy array created from the list of list of training data
    train_data_array = numpy.array(train_data_list)
    print "Finished reading training data"

    # Creating one vs all decision stumps for orientation 0, 90, 180, 270
    print "Generating list of stumps for orientation 0"
    (stumps_0, errors_0) = getStumpsandErrors(train_data_array, array_0, stump_count)
    print "Generating list of stumps for orientation 90"
    (stumps_90, errors_90) = getStumpsandErrors(train_data_array, array_90, stump_count)
    print "Generating list of stumps for orientation 180"
    (stumps_180, errors_180) = getStumpsandErrors(train_data_array, array_180, stump_count)
    print "Generating list of stumps for orientation 270"
    (stumps_270, errors_270) = getStumpsandErrors(train_data_array, array_270, stump_count)

    # Calculate the weights of hypothesis (stumps) using log((1-error)/error)
    error_0_weight_list = [math.log((1 - x) / x) for x in errors_0]
    error_90_weight_list = [math.log((1 - x) / x) for x in errors_90]
    error_180_weight_list = [math.log((1 - x) / x) for x in errors_180]
    error_270_weight_list = [math.log((1 - x) / x) for x in errors_270]

    # Create and return a set of stumps and corresponding weights
    stumpAndError = (stumps_0, stumps_90, stumps_180, stumps_270, error_0_weight_list, \
                     error_90_weight_list, error_180_weight_list, error_270_weight_list)
    return stumpAndError


def classifyImages(test_file, stumpAndError, stump_count):
    adaboost_output = []
    # Get the set of stumps and corresponding weights from the arguments
    (stumps_0, stumps_90, stumps_180, stumps_270, error_0_weight_list, error_90_weight_list, \
     error_180_weight_list, error_270_weight_list) = stumpAndError
    # Read the test data from file
    with open(test_file) as f:
        content = f.readlines()
    print "Reading test data complete."
    print "Beginning Classification"
    confusion_matrix = [[0] * 4 for i in range(4)]
    orientations = [0, 90, 180, 270]
    # Classify based on the max vote from the individual votes
    for line in content:
        actual = orientations.index(int(line.split()[1]))
        row = [int(x) for x in line.split()[2:]]
        vote_0 = getVote(stumps_0, error_0_weight_list, row, stump_count)
        vote_90 = getVote(stumps_90, error_90_weight_list, row, stump_count)
        vote_180 = getVote(stumps_180, error_180_weight_list, row, stump_count)
        vote_270 = getVote(stumps_270, error_270_weight_list, row, stump_count)
        votes = [vote_0, vote_90, vote_180, vote_270]
        # By taking a max from all the votes we get the predicted orientation of the image
        predicted = votes.index(max(votes))
        adaboost_output.append(line.split()[0] + " " + str(orientations[predicted]))
        confusion_matrix[actual][predicted] += 1
    print "Finished Classification"
    adaboost_output_file = open('adaboost_output.txt', 'w')
    for line in adaboost_output:
        adaboost_output_file.write("%s\n" % line)
    # Print the confusion matrix and the accuracy
    print "\nPrinting Confusion Matrix:"
    print "Predicted    0    90   180  270"
    print "Actual                       "
    print "0           " + str(confusion_matrix[0][0]) + "   " + str(confusion_matrix[0][1]) + "   " + str(
        confusion_matrix[0][2]) + "   " + str(confusion_matrix[0][3])
    print "90           " + str(confusion_matrix[1][0]) + "  " + str(confusion_matrix[1][1]) + "   " + str(
        confusion_matrix[1][2]) + "   " + str(confusion_matrix[1][3])
    print "180          " + str(confusion_matrix[2][0]) + "   " + str(confusion_matrix[2][1]) + "  " + str(
        confusion_matrix[2][2]) + "   " + str(confusion_matrix[2][3])
    print "270          " + str(confusion_matrix[3][0]) + "   " + str(confusion_matrix[3][1]) + "   " + str(
        confusion_matrix[3][2]) + "  " + str(confusion_matrix[3][3])
    correct = 0
    total = 0
    for i in range(len(confusion_matrix)):
        correct += confusion_matrix[i][i]
        total += sum(confusion_matrix[i])
    print "Overall Accuracy is (" + str(correct) + "/" + str(total) + "): " + str(
        float(correct) * 100.0 / float(total)) + "%"