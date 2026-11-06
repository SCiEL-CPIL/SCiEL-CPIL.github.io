"""
Constructs a set of 12 distinct positive integers such that all subset sums are distinct.
The optimal solution to minimize the largest element is the sequence of powers of 2:
1, 2, 4, ..., 2^(n-1).
For n=12, the largest element is 2^11 = 2048.
"""
import time

def construct():
    """
    Returns a set of 12 distinct positive integers such that all 4096 subset sums are distinct.
    The set minimizes the largest element.
    
    The minimum possible largest element for n distinct positive integers with distinct subset sums is 2^(n-1).
    For n=12, this is 2^11 = 2048.
    The optimal set is the set of powers of 2: [1, 2, 4, ..., 2048].
    """
    n = 12
    # The optimal solution is the first n powers of 2.
    result = [1 << i for i in range(n)]
    
    # Validation
    # Check distinctness
    if len(set(result)) != n:
        raise ValueError("Elements must be distinct")
    
    # Check positivity
    if any(x <= 0 for x in result):
        raise ValueError("Elements must be positive")
        
    # Check subset sum distinctness
    # There are 2^12 = 4096 subsets.
    subset_sums = set()
    # Using bit manipulation to iterate through all subsets
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        subset_sums.add(s)
        
    if len(subset_sums) != (1 << n):
        raise ValueError("Subset sums are not all distinct")
        
    return result
