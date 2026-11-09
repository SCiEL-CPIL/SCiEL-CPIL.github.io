import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin within 90s wall-clock
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using a vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        sigma[d::d] += d
        
        if d % 2_000_000 == 0:
            if time.time() > deadline - 20:
                break
    
    if time.time() > deadline - 15:
        return [11, 6]

    # 2. Sort indices by sigma value
    # We only care about n >= 1
    # argsort is stable? We just need the values.
    # To save memory, we can sort the slice.
    # sigma[1:] has 10^7 elements.
    
    # Get sorted indices
    # This takes ~1-2 seconds
    try:
        sorted_indices = np.argsort(sigma[1:], kind='mergesort') + 1
        sorted_sigmas = sigma[sorted_indices]
    except:
        return [11, 6]
        
    if time.time() > deadline - 10:
        return [11, 6]

    # 3. Search for best pair
    best_a = 11
    best_b = 6
    best_num = abs(2 * 11 - 3 * 6)  # 4
    best_den = 2 * 6  # 12
    
    # Iterate through groups defined by sorted_sigmas
    n_groups = len(sorted_sigmas)
    i = 0
    while i < n_groups:
        j = i + 1
        while j < n_groups and sorted_sigmas[j] == sorted_sigmas[i]:
            j += 1
        
        # Group is sorted_indices[i:j]
        # Extract nums
        nums = sorted_indices[i:j]
        
        if len(nums) >= 2:
            # Pruning check:
            # Max possible b in this group is max(nums)
            # Min possible error is ~ 1 / (2 * max_b)
            # If best_num / best_den < 1 / (2 * max_b), skip
            max_b = np.max(nums)
            if best_num * (2 * max_b) < best_den:
                # Current best is better than anything this group can offer
                # Note: error = num/den. We compare num/den < 1/(2*max_b)
                # num * 2 * max_b < den
                pass # Skip
            else:
                # Process this group
                # Convert to list for bisect?
                # nums is a numpy array. bisect works on lists.
                # Converting 10^7 elements to list is slow.
                # But we only convert groups.
                # Most groups are small.
                
                nums_list = nums.tolist()
                
                # We want to iterate 'a' such that we find good pairs.
                # Error ~ 1/a. Larger a is better.
                # So iterate nums in descending order?
                # Or just iterate and check.
                
                # To allow early exit if we find a perfect match?
                # We can't exit the whole function easily if we haven't checked all.
                # But we can break the inner loop if we find a very small diff?
                
                for a in nums_list:
                    target = (2.0 * a) / 3.0
                    idx = bisect_left(nums_list, target)
                    
                    candidates = []
                    if idx < len(nums_list):
                        candidates.append(nums_list[idx])
                    if idx > 0:
                        candidates.append(nums_list[idx - 1])
                    
                    for b in candidates:
                        if b == a:
                            continue
                        
                        diff = abs(2 * a - 3 * b)
                        if diff == 0:
                            continue
                        
                        den = 2 * b
                        
                        if diff * best_den < best_num * den:
                            best_num = diff
                            best_den = den
                            best_a = a
                            best_b = b
                            
        i = j
        
        if time.time() > deadline - 5:
            break
            
    return [int(best_a), int(best_b)]
