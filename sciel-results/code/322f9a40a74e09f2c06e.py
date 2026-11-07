import numpy as np

def construct():
    N = 10000000
    
    # 1. Compute sigma(n) for n=1..N
    # Using a sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    # sigma[0] = 0, sigma[n] = sum of divisors
    # Efficient way:
    # for i in range(1, N+1):
    #     sigma[i::i] += i
    # This is O(N log N). Let's see if it's fast enough.
    # N=10^7. Sum of 1/i is ~16.7. So ~1.67 * 10^8 additions.
    # NumPy vectorized addition is fast.
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # 2. Sort indices by sigma value
    # We want to group numbers with the same sigma.
    # Create an array of indices 1..N
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    # Get sigma values for these indices
    sig_vals = sigma[indices]
    
    # Sort indices based on sigma values
    # np.argsort is stable? No, but we don't need stability for grouping, 
    # we just need them sorted by sigma.
    # However, within the same sigma, we want them sorted by value for binary search.
    # If we just sort by sigma, the order within same sigma is arbitrary.
    # We need to sort by (sigma, value).
    # We can create a structured array or use lexsort.
    
    # lexsort: sort by primary key (indices), secondary key (sig_vals)?
    # No, we want primary key sig_vals, secondary key indices.
    # np.lexsort((indices, sig_vals)) sorts by sig_vals first, then indices.
    
    order = np.lexsort((indices, sig_vals))
    
    # Now indices are sorted by sigma, and within same sigma, by value.
    sorted_indices = indices[order]
    sorted_sig_vals = sig_vals[order]
    
    # 3. Iterate through blocks of same sigma
    # Find boundaries where sigma changes
    # diff = np.diff(sorted_sig_vals)
    # changes = np.where(diff != 0)[0]
    # boundaries = np.concatenate(([0], changes + 1, [len(sorted_sig_vals)]))
    
    # But doing this for 10^7 elements creates large arrays.
    # We can iterate in a loop over the sorted array.
    
    best_diff = float('inf')
    best_a = 1
    best_b = 1
    
    # To speed up, we can skip blocks with size 1.
    # We need to find the start and end of each block.
    
    # Efficient way to find block boundaries:
    # We can use np.where(np.diff(sorted_sig_vals) != 0)
    # But let's just iterate.
    
    # Actually, iterating 10^7 times in Python is ~1s.
    # But we need to binary search inside each block.
    
    # Let's try to optimize the inner loop.
    # For each block, we have a sorted array of numbers `nums`.
    # For each `b` in `nums`, find `a` closest to `1.5 * b`.
    
    # We can vectorize the binary search?
    # np.searchsorted(nums, targets, side='left')
    # This is fast.
    
    # So for each block:
    # targets = 1.5 * nums
    # pos = np.searchsorted(nums, targets)
    # Check pos and pos-1
    
    # This is much faster than a Python loop inside the block.
    
    # Let's identify blocks.
    # changes = np.where(np.diff(sorted_sig_vals) != 0)[0] + 1
    # starts = np.concatenate(([0], changes))
    # ends = np.concatenate((changes, [len(sorted_sig_vals)]))
    
    # This creates arrays of size ~10^6 (number of unique sigmas).
    
    diff = np.diff(sorted_sig_vals)
    changes = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_sig_vals)]))
    
    # Iterate over blocks
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
            
        nums = sorted_indices[s:e]
        # nums is sorted
        
        # Calculate targets
        targets = 1.5 * nums.astype(np.float64)
        
        # Find insertion points
        pos = np.searchsorted(nums, targets)
        
        # Candidate positions: pos and pos-1
        # We need to handle boundaries.
        
        # Candidate 1: nums[pos]
        # pos can be out of bounds? searchsorted returns 0..len(nums)
        # If pos == len(nums), nums[pos] is invalid.
        
        # Let's create arrays of candidates
        # cand1_idx = pos
        # cand2_idx = pos - 1
        
        # Filter valid indices
        # valid1 = pos < len(nums)
        # valid2 = pos > 0
        
        # Instead of creating huge arrays, let's just compute diffs for valid ones.
        
        # We want to minimize |nums[i] / nums[j] - 1.5|
        # Which is equivalent to minimizing |nums[i] - 1.5 * nums[j]| / nums[j]
        
        # Let's compute diff for pos
        valid1 = pos < len(nums)
        # pos is an array of indices into nums
        # We need to avoid pos == len(nums)
        
        # Let's just compute for pos and pos-1 where valid.
        
        # To avoid complex masking, we can pad or just use simple logic.
        
        # Let's try a different approach for the inner loop to keep it simple and fast.
        # The number of elements in a block is usually small.
        # If we assume average block size is small (e.g., < 100), we can just loop?
        # No, some blocks are large.
        
        # Let's stick to vectorized searchsorted.
        
        # Candidates from pos
        p1 = pos
        # Clamp p1 to [0, len(nums)-1] for access? No, we check validity.
        
        # Let's compute diffs for p1 where p1 < len(nums)
        # And diffs for p2 = p1 - 1 where p2 >= 0
        
        # We need to ensure a != b.
        # a = nums[p1], b = nums[i].
        # Since targets[i] = 1.5 * nums[i], and nums are sorted,
        # if 1.5 * b is in the array, a might be b?
        # 1.5 * b = b => b = 0. Impossible for b>=1.
        # So a != b is guaranteed if a and b are positive?
        # Wait, if a = 1.5 b, a is different from b.
        # So we don't need to check a != b explicitly if we trust the math.
        # But what if the closest number is b itself?
        # target = 1.5 b. b is in the array.
        # searchsorted(1.5 b) will point to the first element >= 1.5 b.
        # Since b < 1.5 b, the index will be > index of b.
        # So a != b is safe.
        
        # Compute diff for p1
        # We need to handle p1 == len(nums)
        
        # Let's create arrays of candidate values
        # This might be memory heavy if we do it for all blocks at once, but we do it per block.
        
        # For each block:
        # len_block = e - s
        # pos is array of shape (len_block,)
        
        # Valid p1: pos < len_block
        # Valid p2: pos > 0
        
        # Let's just compute the best diff for this block.
        
        # To be safe and simple, if block size is small, loop.
        # If block size is large, vectorize.
        
        if e - s < 100:
            # Loop
            for i in range(len(nums)):
                b = nums[i]
                target = 1.5 * b
                # Binary search
                p = np.searchsorted(nums, target)
                
                # Check p and p-1
                if p < len(nums):
                    a = nums[p]
                    diff_val = abs(a / b - 1.5)
                    if diff_val < best_diff:
                        best_diff = diff_val
                        best_a, best_b = a, b
                if p > 0:
                    a = nums[p-1]
                    diff_val = abs(a / b - 1.5)
                    if diff_val < best_diff:
                        best_diff = diff_val
                        best_a, best_b = a, b
        else:
            # Vectorize
            # This is complex to handle boundaries cleanly in one go.
            # Let's just use the loop for now, but optimize the searchsorted.
            # Actually, the loop above uses np.searchsorted which is C-speed.
            # The Python loop overhead is the bottleneck.
            
            # Let's try to vectorize the diff calculation.
            # We need to check two candidates per b.
            
            # Candidates 1: nums[pos]
            # Candidates 2: nums[pos-1]
            
            # We can compute diffs for all valid ones and take min.
            
            # Let's expand the loop to handle this properly.
            # For now, let's just run the loop version for all, as searchsorted is fast.
            # The loop runs N times total. 10^7 iterations in Python is ~1-2s.
            # Plus searchsorted overhead. Might be ~5-10s.
            # Acceptable.
            
            for i in range(len(nums)):
                b = nums[i]
                target = 1.5 * b
                p = np.searchsorted(nums, target)
                
                if p < len(nums):
                    a = nums[p]
                    diff_val = abs(a / b - 1.5)
                    if diff_val < best_diff:
                        best_diff = diff_val
                        best_a, best_b = a, b
                if p > 0:
                    a = nums[p-1]
                    diff_val = abs(a / b - 1.5)
                    if diff_val < best_diff:
                        best_diff = diff_val
                        best_a, best_b = a, b
                        
    # Check for zero diff
    if best_a / best_b == 1.5:
        # Find next best?
        # This is unlikely.
        pass
        
    return [int(best_a), int(best_b)]
