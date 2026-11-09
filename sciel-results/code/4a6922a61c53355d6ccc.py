import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 500_000 == 0:
            if time.time() > deadline - 30:
                break
    
    # Build groups: dict from sigma value -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
    
    # Now find best pair using two-pointer technique
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group of numbers with same sigma, sort them
    # Then for each b, find a closest to 3b/2
    best_err_num = None  # numerator of error
    best_err_den = None  # denominator of error
    best_pair = None
    best_err_val = float('inf')
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 5:
            break
        nums_sorted = sorted(nums)
        n_len = len(nums_sorted)
        
        # Two-pointer: for each b, the optimal a is near 3b/2
        # Since nums_sorted is sorted, as b increases, 3b/2 increases
        # So we can use a pointer that only moves forward
        ptr = 0
        for i in range(n_len):
            b = nums_sorted[i]
            target = 3 * b // 2  # integer part of 3b/2
            
            # Advance pointer to get close to target
            while ptr < n_len - 1 and nums_sorted[ptr] < target:
                ptr += 1
            
            # Check ptr and ptr-1 (and maybe ptr+1 for safety)
            for j in range(max(0, ptr - 1), min(n_len, ptr + 2)):
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
    
    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
