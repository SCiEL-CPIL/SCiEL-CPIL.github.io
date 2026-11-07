import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for all n up to N using additive sieve
    # This is O(N log N) but highly optimized with NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check time after sieve
    if time.time() - start_time > deadline - 10:
        # Fallback if sieve took too long (unlikely but safe)
        return [int(11), int(6)]

    # 2. Group indices by sigma value
    # We only care about sigma values that appear at least twice
    # Using a dictionary of lists is standard, but we can optimize by 
    # only storing indices.
    
    # To speed up grouping, we can use np.unique on the sigma array? 
    # No, sigma values can be large. 
    # Let's iterate and group. 
    # Optimization: We can filter out sigma values that are "rare" first? 
    # Actually, just building the groups is O(N).
    
    groups = defaultdict(list)
    # Iterating in Python is slow. 
    # Alternative: Sort indices by sigma value?
    # argsort = np.argsort(sigma[1:])
    # sorted_sigma = sigma[1:][argsort]
    # Then find boundaries where sigma changes.
    
    # Let's try the argsort approach which is vectorized and fast.
    indices = np.arange(1, N + 1, dtype=np.int64)
    sigma_vals = sigma[1:]
    
    # Sort indices by their sigma value
    # This is O(N log N) but implemented in C
    sort_idx = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[sort_idx]
    sorted_indices = indices[sort_idx]
    
    # Find boundaries where sigma value changes
    # diff > 0 indicates a new group starts
    diff = np.diff(sorted_sigma)
    # The first element is always a start. 
    # starts = np.where(diff > 0)[0] + 1
    # ends = np.where(diff > 0)[0] + 1 ... plus end
    
    # Simpler: use np.split or just iterate through unique values?
    # np.unique with return_index?
    # unique_sigmas, unique_indices, counts = np.unique(sigma_vals, return_index=True, return_counts=True)
    # This gives the first index. But we need all indices for each sigma.
    
    # The argsort approach is good. Let's extract groups.
    # We can vectorize the extraction of groups?
    # It's hard to vectorize "process each group". 
    # But we can iterate over the groups. 
    # How many groups? Number of unique sigma values. 
    # For N=10^7, unique sigma values are likely < N, probably around 1-2 million?
    # Iterating 1-2 million times in Python is ~1-2 seconds. Acceptable.
    
    # Let's identify group boundaries in the sorted arrays
    change_mask = np.concatenate(([True], diff > 0))
    starts = np.where(change_mask)[0]
    ends = np.concatenate((starts[1:], [len(sorted_indices)]))
    
    # 3. Search for best pair in each group
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate through groups
    # We can prioritize groups with larger size? 
    # Larger groups have more chances to have a pair close to 1.5.
    # But we must check all to be sure? 
    # The error |a/b - 1.5| is minimized. 
    # If we find error = 0, we can stop (but 0 is invalid, so we stop if 2a=3b? No, we skip those).
    # The minimum possible non-zero error depends on the integers.
    
    # Let's sort groups by size descending to find good candidates early?
    # This might help if we want to break early, but we need the GLOBAL min.
    # However, if we find a very small error, can we prune?
    # Not easily.
    
    # Let's just iterate.
    # To speed up, we can skip groups of size 1.
    
    # Pre-compute 1.5 as float? 
    # We want to minimize |a/b - 1.5| = |2a - 3b| / (2b).
    # This is equivalent to minimizing |2a - 3b| / b.
    
    for i in range(len(starts)):
        if time.time() - start_time > deadline - 5:
            break
            
        s_start = starts[i]
        s_end = ends[i]
        group_size = s_end - s_start
        
        if group_size < 2:
            continue
            
        # Extract the sorted numbers in this group
        # They are already sorted because we sorted indices by sigma, 
        # but are they sorted by value? 
        # NO. np.argsort(sigma_vals) sorts by sigma, not by index value.
        # So sorted_indices[s_start:s_end] are the numbers, but NOT sorted by magnitude.
        # We need them sorted by magnitude to use searchsorted for 1.5*b.
        
        nums = sorted_indices[s_start:s_end]
        
        # Sort nums by value
        # This is O(K log K) for each group. Sum of K log K might be slow if many large groups.
        # But most groups are small.
        if group_size > 1:
            nums.sort()
            
        # Now find pairs
        # For each b in nums, find a in nums closest to 1.5 * b.
        
        # Vectorized search:
        # targets = 1.5 * nums
        # indices_a = np.searchsorted(nums, targets)
        
        # We need to check indices_a and indices_a - 1.
        
        targets = 1.5 * nums
        
        # Find insertion points
        # searchsorted returns the index where target would be inserted to keep order.
        # If nums[mid] < target, lo = mid + 1.
        # So indices_a is the first index >= target.
        idx_a = np.searchsorted(nums, targets)
        
        # Candidate indices: idx_a and idx_a - 1
        # We need to handle bounds.
        
        # Let's create arrays of candidate a's.
        # cand1 = nums[idx_a] (where valid)
        # cand2 = nums[idx_a - 1] (where valid)
        
        # Validity checks:
        # 1. idx_a < len(nums) for cand1
        # 2. idx_a - 1 >= 0 for cand2
        # 3. a != b
        # 4. 2*a != 3*b (avoid exact 1.5)
        
        # We can compute error for both sets of candidates and take min.
        
        # Mask for valid idx_a
        valid1 = idx_a < len(nums)
        valid2 = idx_a > 0
        
        # Get candidate a's
        # Be careful with numpy indexing errors.
        
        # For cand1:
        # We need to index nums with idx_a[valid1]
        # And b's are nums[valid1]
        
        # Let's do this in chunks or just vectorized.
        
        # Error formula: |2a - 3b| / (2b)
        # Since we want to minimize this, and b > 0.
        
        # Calculate for cand1
        if np.any(valid1):
            a_cand1 = nums[idx_a[valid1]]
            b_cand1 = nums[valid1]
            
            # Filter a != b
            mask_neq = a_cand1 != b_cand1
            if np.any(mask_neq):
                a_cand1 = a_cand1[mask_neq]
                b_cand1 = b_cand1[mask_neq]
                
                # Filter 2a != 3b
                # 2*a_cand1 != 3*b_cand1
                mask_not_exact = (2 * a_cand1) != (3 * b_cand1)
                if np.any(mask_not_exact):
                    a_cand1 = a_cand1[mask_not_exact]
                    b_cand1 = b_cand1[mask_not_exact]
                    
                    # Calculate error
                    # err = |2a - 3b| / (2b)
                    # Use float64 for precision? 
                    # The task says "score is the binary64 serialization of that rational".
                    # So we should compute the rational exactly, then convert to float64 for comparison?
                    # Or just compute float error?
                    # "Factorization and rational error are exact; the score is the binary64 serialization of that rational"
                    # This implies we should compute the exact rational error, then cast to float64 for the final score.
                    # But for comparison during search, float64 might be sufficient if we are careful, 
                    # or we can compare fractions.
                    # Comparing fractions |2a-3b|/2b is equivalent to comparing |2a-3b|*2*b_best vs ...
                    # Let's stick to float64 for speed, as differences will likely be distinguishable.
                    # If we find a very small error, we can re-verify with exact rational.
                    
                    err_cand1 = np.abs(2 * a_cand1 - 3 * b_cand1) / (2 * b_cand1)
                    
                    min_idx = np.argmin(err_cand1)
                    min_err = err_cand1[min_idx]
                    
                    if min_err < best_error:
                        best_error = min_err
                        best_a = int(a_cand1[min_idx])
                        best_b = int(b_cand1[min_idx])
                        
                        # Optional: if best_error is very small, we might break early?
                        # But we need the global min.
        
        # Calculate for cand2
        if np.any(valid2):
            a_cand2 = nums[idx_a[valid2] - 1]
            b_cand2 = nums[valid2]
            
            mask_neq = a_cand2 != b_cand2
            if np.any(mask_neq):
                a_cand2 = a_cand2[mask_neq]
                b_cand2 = b_cand2[mask_neq]
                
                mask_not_exact = (2 * a_cand2) != (3 * b_cand2)
                if np.any(mask_not_exact):
                    a_cand2 = a_cand2[mask_not_exact]
                    b_cand2 = b_cand2[mask_not_exact]
                    
                    err_cand2 = np.abs(2 * a_cand2 - 3 * b_cand2) / (2 * b_cand2)
                    
                    min_idx = np.argmin(err_cand2)
                    min_err = err_cand2[min_idx]
                    
                    if min_err < best_error:
                        best_error = min_err
                        best_a = int(a_cand2[min_idx])
                        best_b = int(b_cand2[min_idx])

    # Final check
    if best_error == float('inf'):
        return [int(11), int(6)]
        
    return [int(best_a), int(best_b)]
