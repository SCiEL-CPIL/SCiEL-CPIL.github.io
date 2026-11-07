import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for all n <= N
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() - start_time > deadline - 40:
            break
    
    if time.time() - start_time > deadline - 30:
        return [int(11), int(6)]
    
    # 2. Group numbers by sigma value
    # Using a dictionary mapping sigma value to list of numbers
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() - start_time > deadline - 15:
        return [int(11), int(6)]
    
    # Sort the lists in each group to allow binary search
    for key in groups:
        groups[key].sort()
        
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # 3. Search for pairs
    # Iterate b from N down to 1 to find large b first (smaller error for same diff)
    for b in range(N, 0, -1):
        if time.time() - start_time > deadline:
            break
            
        s_b = sigma[b]
        nums = groups[s_b]
        
        if len(nums) < 2:
            continue
            
        # We want a such that a/b approx 3/2 => a approx 1.5*b
        target = 1.5 * b
        
        # Binary search for target in nums
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        
        # Check candidates around lo
        # lo is the first index such that nums[lo] >= target
        # So we check lo and lo-1
        for idx in [lo, lo - 1]:
            if 0 <= idx < len(nums):
                a = nums[idx]
                if a == b:
                    continue
                # Check if ratio is exactly 3/2 (invalid)
                if 2 * a == 3 * b:
                    continue
                
                # Calculate error exactly
                # err = |a/b - 3/2| = |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue # Invalid
                
                err = diff / (2 * b)
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
                    # If we found a diff of 1, this is the best possible error for this b.
                    # Since we are iterating b downwards, any future b' < b will have
                    # error >= 1/(2b') > 1/(2b) = err (assuming min diff 1).
                    # So we can stop if diff == 1.
                    if diff == 1:
                        return [int(best_a), int(best_b)]
                        
    if best_error == float('inf'):
        return [int(11), int(6)]
        
    return [int(best_a), int(best_b)]
