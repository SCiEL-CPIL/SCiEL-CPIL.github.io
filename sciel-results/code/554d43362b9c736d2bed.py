import time
import math
import bisect
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # The loop is the bottleneck, but vectorized ops help
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() - start_time > deadline:
        return [11, 6]

    # 2. Find groups with count >= 2
    vals = sigma[1:]
    
    # Use np.unique to group
    # return_inverse gives the index in the unique array for each element
    # return_counts gives the count of each unique value
    unique_vals, inverse, counts = np.unique(vals, return_inverse=True, return_counts=True)
    
    # Find indices of unique values that have count >= 2
    valid_unique_indices = np.where(counts >= 2)[0]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Process each valid group
    for u_idx in valid_unique_indices:
        if time.time() - start_time > deadline:
            break
            
        # Get the original indices (0-based in vals, so 1-based in n) for this group
        # inverse == u_idx gives the positions in vals
        # We need the actual numbers n
        # np.where(inverse == u_idx)[0] gives the indices in vals
        # This creates a boolean mask or array of indices.
        # For large N, this might be slow if done many times.
        # However, the sum of sizes of all groups is N.
        # So total work is O(N) for extracting indices.
        
        # More efficient: 
        # We can pre-compute the groups? No, memory.
        # Let's just do it.
        
        group_positions = np.where(inverse == u_idx)[0]
        if len(group_positions) < 2:
            continue
            
        # Convert to 1-based numbers
        nums = group_positions + 1
        nums.sort()
        
        # Find best pair in nums
        # Same logic as before
        for b in nums:
            target = 1.5 * b
            idx_b = bisect.bisect_left(nums, target)
            
            candidates = []
            if idx_b < len(nums):
                candidates.append(nums[idx_b])
            if idx_b > 0:
                candidates.append(nums[idx_b - 1])
                
            for a in candidates:
                if a == b:
                    continue
                err = abs(a / b - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b

    return [int(best_a), int(best_b)]
