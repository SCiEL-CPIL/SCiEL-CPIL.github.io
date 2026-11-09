import time
import numpy as np
from collections import defaultdict
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 2_000_000 == 0:
            if time.time() > deadline - 40:
                break
    
    # Build groups: dict from sigma value -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
    
    # Now find best pair using bisect for O(log n) lookup
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group of numbers with same sigma, sort them
    # Then for each b, find a closest to 3b/2 using binary search
    best_err_num = None  # numerator of error
    best_err_den = None  # denominator of error
    best_pair = None
    best_err_val = float('inf')
    
    # Sort groups by size descending to process larger groups first
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s, nums in sorted_groups:
        if len(nums) < 2:
            continue
        if time.time() > deadline - 10:
            break
        nums_sorted = sorted(nums)
        n_len = len(nums_sorted)
        
        # For each b, use bisect to find closest a to 3b/2
        for i in range(n_len):
            b = nums_sorted[i]
            target = 3 * b / 2  # use float for bisect
            
            # Find insertion point
            pos = bisect_left(nums_sorted, target)
            
            # Check candidates around pos
            for j in range(max(0, pos - 1), min(n_len, pos + 2)):
                a = nums_sorted[j]
                if a == b:
                    continue
                # err = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # error is exactly 0, invalid
                den = 2 * b
                err_val = num / den
                if err_val < best_err_val:
                    best_err_val = err_val
                    best_err_num = num
                    best_err_den = den
                    best_pair = (a, b)
            
            # Time check in inner loop
            if i % 1000 == 0 and time.time() > deadline - 10:
                break
    
    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
