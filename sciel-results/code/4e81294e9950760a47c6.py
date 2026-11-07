def construct():
    # The maximum size of such a set is 31.
    # A valid set is the range of integers from 226 to 256 inclusive.
    # This set satisfies the condition that for any k, the sum of the k smallest
    # elements is strictly greater than the sum of the k-1 largest elements.
    # This ensures that subset sums of different cardinalities are disjoint.
    
    optimal_set = list(range(226, 257))
    
    return optimal_set
