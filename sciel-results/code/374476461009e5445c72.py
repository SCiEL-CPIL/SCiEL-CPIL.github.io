import numpy as np
import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # 1. Compute sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 50:
        # Fallback to a known pair if sieve took too long (unlikely)
        return [23, 15]

    # 2. Group by sigma
    sigma_map = defaultdict(list)
    # Using a Python loop to populate the dict
    # This might be slow, but 10M iterations is ~1s
    for n in range(1, N + 1):
        sigma_map[sigma[n]].append(n)

    elapsed = time.time() - start_time
    if elapsed > 55:
        return [23, 15]

    # 3. Search for best pair
    best_a, best_b = 23, 15
    best_err = abs(23/15 - 1.5)
    
    # Iterate through groups
    for sig_val, nums in sigma_map.items():
        if len(nums) < 2:
            continue
        
        # Sort the list of numbers with this sigma
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5*b
        # We can iterate b and binary search a
        for b in nums:
            target_a = 1.5 * b
            # Find insertion point
            pos = bisect.bisect_left(nums, target_a)
            
            # Check candidates around pos
            # pos-1, pos, pos+1
            candidates = []
            if pos > 0:
                candidates.append(nums[pos - 1])
            if pos < len(nums):
                candidates.append(nums[pos])
            if pos + 1 < len(nums):
                candidates.append(nums[pos + 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Calculate error
                # |a/b - 1.5|
                # To avoid float precision issues, use rational comparison?
                # Error = |2a - 3b| / (2b)
                # We want to minimize this.
                # Compare |2a - 3b| / b  vs best_err * 2?
                # Let's just use float for simplicity, N is 1e7, precision is fine.
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b
                    
    # Final check
    elapsed = time.time() - start_time
    # Return as Python ints
    return [int(best_a), int(best_b)]
