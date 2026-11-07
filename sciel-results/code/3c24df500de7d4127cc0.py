import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n) using a fast sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Vectorized approach: for each divisor d, add d to all multiples of d
    # This is O(N log N) but with much better constants in NumPy
    for d in range(1, N + 1):
        if time.time() > deadline - 30.0:
            return [23, 15]
        # Add d to sigma[d], sigma[2d], sigma[3d], ...
        sigma[d::d] += d
        
    if time.time() > deadline - 20.0:
        return [23, 15]

    # 2. Group numbers by sigma value
    # Use a more memory-efficient approach: sort by sigma value
    # Create array of (sigma_value, n) pairs and sort by sigma_value
    indices = np.arange(1, N + 1, dtype=np.int32)
    sigma_vals = sigma[1:N+1].copy()
    
    # Sort indices by sigma value
    sort_order = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[sort_order]
    sorted_n = indices[sort_order]
    
    if time.time() > deadline - 10.0:
        return [23, 15]

    # 3. Find groups of equal sigma values and search for best pair
    best_error = float('inf')
    best_pair = (23, 15)
    
    # Find boundaries where sigma value changes
    # Use diff to find where sorted_sigma changes
    changes = np.concatenate(([True], sorted_sigma[1:] != sorted_sigma[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate([group_starts[1:], [len(sorted_n)]])
    
    # Iterate through groups
    for start, end in zip(group_starts, group_ends):
        if time.time() > deadline - 5.0:
            break
            
        group_len = end - start
        if group_len < 2:
            continue
            
        nums = sorted_n[start:end]
        
        # For each b in nums, find a in nums such that a/b is close to 1.5
        # a approx 1.5 * b
        # Check a window of candidates around the target position
        for i, b in enumerate(nums):
            target = 1.5 * b
            idx = bisect_left(nums, target)
            
            # Check a few candidates around the target
            # This handles cases where the closest element isn't exactly at idx
            check_positions = set()
            for offset in range(-3, 4):
                pos = idx + offset
                if 0 <= pos < group_len:
                    check_positions.add(pos)
            
            for pos in check_positions:
                a = int(nums[pos])
                b_val = int(b)
                if a == b_val:
                    continue
                # Calculate error |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b_val)
                if num == 0:
                    continue
                
                denom = 2.0 * b_val
                err = num / denom
                
                if err < best_error:
                    best_error = err
                    best_pair = (a, b_val)
                    
    return [int(best_pair[0]), int(best_pair[1])]
