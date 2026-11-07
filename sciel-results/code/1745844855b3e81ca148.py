import time
import math
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve
    # This is O(N log N) and typically takes a few seconds for N=10^7
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    if time.time() - start_time > deadline - 10:
        return [11, 6]
        
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() - start_time > deadline - 5:
        return [11, 6]
        
    # 2. Build mapping from sigma value to list of numbers
    # We only need to consider numbers that could potentially form a good ratio.
    # However, to be safe, we include all.
    sigma_to_nums = defaultdict(list)
    
    # Vectorized-ish loop for grouping is tricky, so we do a Python loop.
    # To speed up, we can filter out some obvious non-candidates if needed, 
    # but let's try full first.
    for n in range(1, N + 1):
        s = int(sigma[n])
        sigma_to_nums[s].append(n)
        
    # Sort the lists in each group for binary search
    # This is expensive but necessary for efficient neighbor lookup
    for s_val, nums in sigma_to_nums.items():
        nums.sort()
        
    if time.time() - start_time > deadline - 2:
        return [11, 6]

    # 3. Search for best pair
    best_error_num = float('inf')
    best_error_den = 1
    best_a = 11
    best_b = 6
    
    # We iterate through b. 
    # To minimize |a/b - 1.5|, we want a ~ 1.5 b.
    # So b should be at least 2 (for a>=3).
    # Also a <= N, so 1.5 b <= N => b <= N/1.5 = 2N/3.
    
    max_b = int(N * 2 / 3)
    
    # To improve speed, we can sample b or check all.
    # Checking all b with binary search in the group is O(N log K).
    # N=10^7, log K is small. This should be feasible in ~10-20s.
    
    for b in range(1, max_b + 1):
        if time.time() - start_time > deadline:
            break
            
        s_val = int(sigma[b])
        nums = sigma_to_nums.get(s_val)
        if not nums or len(nums) < 2:
            continue
            
        target_a = 1.5 * b
        
        # Find index in nums where target_a would be inserted
        # bisect_left returns the first index where nums[idx] >= target_a
        idx = bisect.bisect_left(nums, target_a)
        
        # Check neighbors: idx-2 to idx+2
        # We want to check a few candidates on both sides
        start_idx = max(0, idx - 3)
        end_idx = min(len(nums), idx + 4)
        
        for i in range(start_idx, end_idx):
            a = nums[i]
            if a == b:
                continue
            if a > N:
                continue
            # Check if 2a == 3b (exact 3/2 is invalid)
            if 2 * a == 3 * b:
                continue
                
            # Calculate error: |a/b - 3/2| = |2a - 3b| / (2b)
            num = abs(2 * a - 3 * b)
            den = 2 * b
            
            # Compare num/den < best_error_num / best_error_den
            # num * best_error_den < best_error_num * den
            if num * best_error_den < best_error_num * den:
                best_error_num = num
                best_error_den = den
                best_a = a
                best_b = b
                
    # Return Python ints
    return [int(best_a), int(best_b)]
