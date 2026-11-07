import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N
    # Using a vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [1, 1]

    # 2. Bucket indices by sigma value
    # We can use a dictionary mapping sigma value to list of n's
    buckets = defaultdict(list)
    # To speed up iteration, we can iterate over the numpy array
    # But converting to Python loop for bucketing is O(N)
    for i in range(1, N + 1):
        buckets[int(sigma[i])].append(i)
        
    if time.time() > deadline:
        # Fallback: find any pair
        for key, lst in buckets.items():
            if len(lst) >= 2:
                return [lst[0], lst[1]]
        return [1, 1]

    # 3. Search for best ratio
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for key, lst in buckets.items():
        if len(lst) < 2:
            continue
            
        # Sort the list
        lst.sort()
        
        # For each b in lst, find a in lst closest to 1.5 * b
        for b in lst:
            if time.time() > deadline:
                break
                
            target = 1.5 * b
            idx = bisect.bisect_left(lst, target)
            
            # Check neighbors
            # We want a != b
            # Candidates are at idx and idx-1
            if idx < len(lst):
                a = lst[idx]
                if a != b:
                    # Error is |a/b - 1.5| = |a - 1.5b|/b = |2a - 3b| / (2b)
                    # Using float for comparison
                    err = abs(2 * a - 3 * b) / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
            if idx > 0:
                a = lst[idx - 1]
                if a != b:
                    err = abs(2 * a - 3 * b) / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        
        if time.time() > deadline:
            break

    return [int(best_a), int(best_b)]
