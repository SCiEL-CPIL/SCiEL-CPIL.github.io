import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 10_000_000
    
    # Fast additive sieve using NumPy vectorized operations
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 100000 == 0 and time.time() - start_time > deadline - 40:
            break
    
    if time.time() - start_time > deadline - 30:
        return [int(11), int(6)]
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() - start_time > deadline - 15:
        return [int(11), int(6)]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Sort groups by size descending - larger groups more likely to have good pairs
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s_val, nums in sorted_groups:
        if time.time() - start_time > deadline:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        
        for b in nums:
            if time.time() - start_time > deadline:
                break
            
            target = 3 * b // 2
            
            # Binary search for closest to target
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            
            for idx in [lo, lo - 1]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a == b:
                        continue
                    if 2 * a == 3 * b:
                        continue
                    err = abs(2 * a - 3 * b) / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
    
    if best_error == float('inf'):
        return [int(11), int(6)]
    return [int(best_a), int(best_b)]
