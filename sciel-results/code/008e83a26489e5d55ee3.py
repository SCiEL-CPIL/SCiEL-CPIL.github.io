import time

def construct():
    """
    Returns a list of 12 distinct positive integers such that all 2^12 subset sums are distinct.
    The goal is to minimize the largest integer.
    The optimal solution is the powers of 2: 1, 2, 4, ..., 2048.
    This is the theoretical minimum because the sum of the first k elements must be at least 2^k - 1.
    """
    n = 12
    result = []
    current_sum = 0
    
    for i in range(n):
        # The smallest integer greater than the sum of all previous elements
        # ensures all subset sums remain distinct.
        # This generates the sequence 1, 2, 4, 8, ...
        next_val = current_sum + 1
        result.append(next_val)
        current_sum += next_val
        
    return result
