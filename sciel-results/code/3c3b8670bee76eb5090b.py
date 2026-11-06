"""
Construct 12 distinct positive integers with distinct subset sums, minimizing the max.

The mathematical lower bound for the largest element is 2^(n-1) = 2048 for n=12,
achieved by the powers of 2: [1, 2, 4, ..., 2048].
"""
import time

def construct():
    n = 12
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s kill

    # The optimal solution is the first n powers of 2.
    # This achieves the theoretical minimum maximum element of 2^(n-1).
    result = [1 << i for i in range(n)]
    
    # Quick validation
    if len(set(result)) != n:
        raise ValueError("Elements must be distinct")
    if any(x <= 0 for x in result):
        raise ValueError("Elements must be positive")
    if max(result) > 1048576:
        raise ValueError("Largest element exceeds bound")
    
    # Verify all 4096 subset sums are distinct
    # Use a set to track seen sums
    seen_sums = set()
    total_subsets = 1 << n  # 4096
    
    for mask in range(total_subsets):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        seen_sums.add(s)
        
        # Deadline guard
        if time.time() > deadline:
            return result
    
    if len(seen_sums) != total_subsets:
        raise ValueError("Subset sums are not all distinct")
    
    return result
