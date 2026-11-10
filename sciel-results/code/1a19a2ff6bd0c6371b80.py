import time
import sys
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
MAX_N = 10_000_000
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)
        
    N = MAX_N
    
    # Step 1: Compute sigma(n) for all n up to N using NumPy sieve
    sigma_np = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for sigma
    # Complexity: O(N log N)
    for i in range(1, N + 1):
        if time_left() < 5:
            # Fallback if sigma computation is too slow
            return [11, 6]
        sigma_np[i::i] += i
        
    if time_left() < 2:
        return [11, 6]
        
    best_err = float('inf')
    best_pair = (11, 6) # Fallback: sigma(11)=12, sigma(6)=12, err=|11/6-1.5|=0.333
    
    # Step 2: Group indices by sigma value
    # We use a dictionary mapping sigma value to a list of n
    # To save memory and time, we only care about groups with size > 1
    # But we don't know sizes until we group them.
    # We can use a dict of lists.
    
    # Optimization: Instead of storing all lists, we can store them in a dict.
    # The number of unique sigma values is much less than N.
    
    fibers = {}
    
    # Iterate through n and add to fibers
    # This is a Python loop, which might be slow for 10^7.
    # We can vectorize the grouping?
    # np.unique with return_inverse?
    # That gives unique values and indices.
    # Then we can bincount or similar to get groups.
    
    # Let's try np.unique approach
    unique_sigmas, inverse_indices = np.unique(sigma_np[1:N+1], return_inverse=True)
    
    # Now inverse_indices[i] is the index of sigma_np[i+1] in unique_sigmas
    # We need to group indices by inverse_indices.
    
    # We can use np.argsort on inverse_indices to group them together
    sorted_indices = np.argsort(inverse_indices, kind='stable')
    
    # Now, indices with the same sigma are contiguous in sorted_indices
    # We need to find the boundaries of these groups.
    
    # Find where the group changes
    # diff = np.diff(inverse_indices[sorted_indices])
    # changes = np.where(diff != 0)[0]
    
    # This might be memory intensive (10^7 ints).
    # Let's check memory. 10^7 * 8 bytes = 80MB. Acceptable.
    
    if time_left() < 2:
        return [11, 6]
        
    # Get the sorted sigma values
    sorted_sigmas = inverse_indices[sorted_indices]
    
    # Find boundaries
    # We add a sentinel at the end
    # boundaries = np.where(np.diff(sorted_sigmas) != 0)[0] + 1
    # start_indices = np.concatenate(([0], boundaries))
    # end_indices = np.concatenate((boundaries, [len(sorted_indices)]))
    
    # Actually, we can just iterate through the sorted array and process groups.
    # But Python loop over 10^7 items is slow.
    # We can vectorize the group processing?
    
    # Alternative:
    # For each unique sigma, get the list of n's.
    # This requires inverting the mapping.
    
    # Let's stick to the vectorized group finding.
    # But we only care about groups with size >= 2.
    
    # Let's find the sizes of the groups.
    # We can use np.diff on the unique values?
    # No, we have the flat array.
    
    # Let's use a loop over the unique sigmas, but only if there are many small groups?
    # No, the number of unique sigmas is around N / log N ?
    # Actually, sigma(n) can be large.
    
    # Let's try a different approach for grouping:
    # Use a dictionary, but fill it using NumPy to avoid Python loop?
    # It's hard to fill a dict of lists with NumPy directly.
    
    # Let's go back to the Python loop for grouping, but optimize it.
    # Or, use the sorted_indices and process contiguous blocks.
    
    # We can find the start and end of each block using vectorized operations.
    # But then we still need to process each block.
    
    # Let's assume the number of blocks is manageable (e.g., < 10^6).
    # Then we can loop over the blocks.
    
    # Find block boundaries
    # We compare adjacent elements in sorted_sigmas
    # If they are different, a new block starts.
    
    # To avoid huge memory allocation for diff, we can do it in chunks?
    # Or just use np.diff, it's 80MB.
    
    if time_left() < 1:
        return [11, 6]
        
    # Create a mask for where the value changes
    # We need to handle the first element.
    # changes = np.ones(len(sorted_sigmas), dtype=bool)
    # changes[1:] = sorted_sigmas[1:] != sorted_sigmas[:-1]
    # start_indices = np.where(changes)[0]
    
    # This is also 80MB.
    
    # Let's try this.
    
    changes = np.ones(len(sorted_sigmas), dtype=bool)
    changes[1:] = sorted_sigmas[1:] != sorted_sigmas[:-1]
    start_indices = np.where(changes)[0]
    
    # End indices are start_indices[1:] and the last one is len
    end_indices = np.concatenate((start_indices[1:], [len(sorted_sigmas)]))
    
    # Now we have start_indices and end_indices for each group.
    # We iterate over these groups.
    
    # Number of groups is len(start_indices)
    # We can filter out groups of size 1.
    
    # We want to process groups with size >= 2.
    
    # To speed up, we can check the size first.
    sizes = end_indices - start_indices
    
    # Only process if size >= 2
    valid_group_mask = sizes >= 2
    
    # Get the start and end indices for valid groups
    valid_starts = start_indices[valid_group_mask]
    valid_ends = end_indices[valid_group_mask]
    
    # Iterate over valid groups
    # We can vectorize the inner search?
    # For each group, we have a slice of sorted_indices.
    # The numbers are n = sorted_indices[start:end] + 1 (since sigma_np is 1-indexed in our array, but indices are 0-based for the slice)
    # Wait, sigma_np[i] corresponds to n=i.
    # We created sigma_np of size N+1.
    # unique_sigmas was from sigma_np[1:N+1].
    # So index 0 in inverse_indices corresponds to n=1.
    # So n = sorted_indices[start:end] + 1.
    
    # We need to find best pair in each group.
    # We can do this in a loop.
    
    # If there are too many groups, this loop might be slow.
    # But most groups are small.
    
    # Let's add a time check inside the loop.
    
    for i in range(len(valid_starts)):
        if time_left() < 1:
            break
            
        start = valid_starts[i]
        end = valid_ends[i]
        
        # Get the numbers in this group
        # They are already sorted because sorted_indices is sorted by inverse_indices (sigma), and within the same sigma, the order is stable (original index order).
        # Wait, np.argsort is stable, so within the same sigma, the numbers are in increasing order of n.
        # So the slice is sorted.
        
        nums = sorted_indices[start:end] + 1
        
        # nums is a numpy array of int64, sorted.
        
        # For each a in nums, find best b.
        # We can vectorize this?
        # For each a, we want to find b in nums closest to 2a/3.
        
        # targets = (2 * nums) / 3.0
        # We need to find the index in nums for each target.
        # np.searchsorted(nums, targets) gives the index where target would be inserted to maintain order.
        # Let idx = np.searchsorted(nums, targets)
        # Then the candidate b's are nums[idx-1] and nums[idx] (if in bounds).
        
        # We can do this for all a in the group at once!
        
        targets = (2.0 * nums) / 3.0
        
        # searchsorted
        idx = np.searchsorted(nums, targets)
        
        # We need to check idx-1 and idx.
        # We need to handle bounds.
        
        # b1 = nums[idx-1]
        # b2 = nums[idx]
        
        # We create arrays for b1 and b2, handling out-of-bounds.
        
        # b1 indices
        idx1 = idx - 1
        valid1 = idx1 >= 0
        b1 = np.empty(len(nums), dtype=np.int64)
        b1[valid1] = nums[idx1[valid1]]
        # For invalid ones, we can set them to -1 or 0, and filter later.
        b1[~valid1] = -1
        
        # b2 indices
        idx2 = idx
        valid2 = idx2 < len(nums)
        b2 = np.empty(len(nums), dtype=np.int64)
        b2[valid2] = nums[idx2[valid2]]
        b2[~valid2] = -1
        
        # Now we have candidate b1 and b2 for each a.
        # We need to calculate error for (a, b1) and (a, b2).
        # Error = |a/b - 1.5|
        
        # We must exclude b == a.
        
        # Calculate errors for b1
        # valid_b1 = (b1 != -1) & (b1 != nums)
        # err1 = np.abs(nums / b1[valid_b1] - 1.5)
        
        # This might have division by zero if b1 is 0? No, b1 >= 1.
        # But if b1 == -1, we skip.
        
        # Let's combine b1 and b2 into one array of candidates?
        # Or just process them separately.
        
        # Process b1
        mask_b1 = (b1 != -1) & (b1 != nums)
        if np.any(mask_b1):
            a_vals = nums[mask_b1]
            b_vals = b1[mask_b1]
            errs = np.abs(a_vals / b_vals - 1.5)
            
            # Find min error
            min_idx = np.argmin(errs)
            min_err = errs[min_idx]
            
            if min_err < best_err:
                best_err = min_err
                a_best = int(a_vals[min_idx])
                b_best = int(b_vals[min_idx])
                
                # Check if error is exactly 0
                # If so, we shouldn't update best if best is already non-zero?
                # No, we want the minimum non-zero error.
                # If min_err is 0, we skip this update?
                # But there might be another pair in this group with small non-zero error.
                # So we should not just take the min. We should find the min non-zero error.
                
                # Let's filter out zeros.
                non_zero_mask = errs > 0
                if np.any(non_zero_mask):
                    nz_errs = errs[non_zero_mask]
                    nz_idx = np.where(non_zero_mask)[0]
                    min_nz_idx = nz_idx[np.argmin(nz_errs)]
                    min_nz_err = nz_errs[np.argmin(nz_errs)]
                    
                    if min_nz_err < best_err:
                        best_err = min_nz_err
                        a_best = int(a_vals[min_nz_idx])
                        b_best = int(b_vals[min_nz_idx])
                        best_pair = (a_best, b_best)
                        
        # Process b2
        mask_b2 = (b2 != -1) & (b2 != nums)
        if np.any(mask_b2):
            a_vals = nums[mask_b2]
            b_vals = b2[mask_b2]
            errs = np.abs(a_vals / b_vals - 1.5)
            
            non_zero_mask = errs > 0
            if np.any(non_zero_mask):
                nz_errs = errs[non_zero_mask]
                nz_idx = np.where(non_zero_mask)[0]
                min_nz_idx = nz_idx[np.argmin(nz_errs)]
                min_nz_err = nz_errs[np.argmin(nz_errs)]
                
                if min_nz_err < best_err:
                    best_err = min_nz_err
                    a_best = int(a_vals[min_nz_idx])
                    b_best = int(b_vals[min_nz_idx])
                    best_pair = (a_best, b_best)
                    
    # Verify result
    a_final, b_final = best_pair
    if not (1 <= a_final <= N and 1 <= b_final <= N):
        return [11, 6]
    if a_final == b_final:
        return [11, 6]
    if sigma_np[a_final] != sigma_np[b_final]:
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - Fraction(3, 2))
    if err_final == 0:
        return [11, 6]
        
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
