def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    The optimal set is the powers of 2: 1, 2, 4, ..., 2^(n-1).
    This implementation constructs the set iteratively by selecting the smallest
    possible next element that maintains the property that all subset sums are distinct.
    For a set of distinct positive integers with distinct subset sums, the elements
    must be superincreasing: a_k > sum(a_1, ..., a_{k-1}).
    The smallest such sequence starting with 1 is 1, 2, 4, ...
    """
    n = 12
    elements = []
    current_sum = 0
    
    for _ in range(n):
        # The next element must be strictly greater than the sum of all previous elements
        # to ensure that all subset sums involving the new element are distinct from
        # all subset sums of the previous elements.
        next_val = current_sum + 1
        elements.append(next_val)
        current_sum += next_val
        
    return elements
