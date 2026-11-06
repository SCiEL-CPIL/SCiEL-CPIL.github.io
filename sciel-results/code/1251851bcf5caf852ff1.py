def construct():
    """
    Constructs a set of n=12 distinct positive integers such that all 
    subset sums are distinct, minimizing the largest integer.
    
    The optimal strategy is to use a superincreasing sequence where each 
    element is strictly greater than the sum of all previous elements. 
    To minimize the largest element, we choose the smallest possible 
    integer at each step that satisfies this condition.
    
    Let S be the sum of the elements chosen so far.
    The next element a_k must satisfy a_k > S.
    The smallest such integer is S + 1.
    
    This generates the sequence 1, 2, 4, 8, ..., which are powers of 2.
    """
    n = 12
    result = []
    current_sum = 0
    
    for _ in range(n):
        # The smallest integer that preserves distinct subset sums
        # is one greater than the sum of all previous elements.
        next_val = current_sum + 1
        result.append(next_val)
        current_sum += next_val
        
    return result
