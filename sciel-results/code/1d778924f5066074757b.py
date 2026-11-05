import math
import time
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 120.0  # seconds
    MAX_S = 10**12
    TIME_MARGIN = 10.0 # Reserve time for final check and safety

    # 1. Generate primes up to 10^6 for trial division / sieve limits
    # We need primes up to sqrt(MAX_S) = 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Segmented Sieve for Phi using NumPy for vectorization
    # Computes phi for all n in [start, end]
    def segmented_phi(start, end):
        size = end - start + 1
        if size <= 0:
            return []
        
        # Initialize nums and phis as numpy arrays
        # nums[i] holds the current value of the number (start+i) after removing prime factors
        # phis[i] holds the current value of phi(start+i)
        nums = np.arange(start, end + 1, dtype=np.int64)
        phis = nums.copy()
        
        # Sieve with primes up to sqrt(end)
        for p in primes:
            if p * p > end:
                break
                
            # Find the first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            if first_mult > end:
                continue
                
            # Create indices of multiples of p in [start, end]
            # We operate on the slice directly to avoid creating large index arrays if possible, 
            # but numpy slicing with step is efficient.
            # However, we need to check divisibility of the *remaining* part in nums.
            
            # Get the slice of nums corresponding to multiples of p
            # start_idx is 0-based index in the block
            start_idx = first_mult - start
            # The indices are start_idx, start_idx + p, ...
            
            # We need to update nums and phis for these indices.
            # Instead of fancy indexing which creates copies, we can iterate or use masks.
            # Given the vectorization goal, let's use the slice view if possible, 
            # but step > 1 views are not always contiguous or easily writable in a way that 
            # supports element-wise operations without intermediate arrays in older numpy versions.
            # Let's stick to the efficient mask approach but optimize the inner loop.
            
            # Create an array of indices
            idxs = np.arange(start_idx, size, p)
            
            # Check which of these are actually divisible by p in the *current* nums state
            # Note: nums[idxs] % p == 0 is the correct check.
            mask = nums[idxs] % p == 0
            
            if not np.any(mask):
                continue
            
            # Get the indices that need updating
            update_idxs = idxs[mask]
            
            # Update phi: phis[idx] -= phis[idx] // p
            phis[update_idxs] -= phis[update_idxs] // p
            
            # Remove all factors of p from nums[update_idxs]
            # This part is the bottleneck. We do it in a loop for each prime.
            # Since we only need to do this for primes up to sqrt(end), and most numbers
            # have few small prime factors, this should be okay.
            # Vectorized removal:
            # We divide by p repeatedly until not divisible.
            # This is hard to vectorize perfectly without a loop over the number of divisions.
            # However, we can do:
            # while any(nums[update_idxs] % p == 0):
            #     nums[update_idxs] //= p
            
            # Optimization: Only a small fraction of numbers have p^2 as a factor.
            # Let's just do the division once, then check for remainder.
            # Actually, the standard segmented sieve approach for phi requires removing ALL factors.
            
            # Let's try a vectorized loop for the removal.
            # Since p is fixed for this iteration:
            current_nums = nums[update_idxs].copy()
            while True:
                div_mask = current_nums % p == 0
                if not np.any(div_mask):
                    break
                current_nums[div_mask] //= p
            nums[update_idxs] = current_nums
            
        # Handle remaining prime factors > sqrt(end)
        # If nums[idx] > 1, it is a prime factor (or 1)
        remaining = nums > 1
        if np.any(remaining):
            rem_idxs = np.where(remaining)[0]
            # For these, nums[idx] is the remaining prime factor P.
            # phi(n) = phi(n) * (P-1) / P
            # Since P is prime, P-1 is not divisible by P (unless P=2, but P>sqrt(end)>=2, so P!=2? 
            # Wait, if end < 4, sqrt(end) < 2. But end is large. 
            # If P=2, it would have been processed in the loop. So P is odd prime or P=1? No, P>1.
            # So P is an odd prime.
            phis[rem_idxs] = phis[rem_idxs] - phis[rem_idxs] // nums[rem_idxs]
                
        return phis.tolist()

    # 3. Helper to find longest run of distinct values
    def find_longest_distinct_run(phis, block_start):
        seen_vals = {}  # value -> index
        l = 0
        max_len = 0
        max_start_idx = 0
        
        for r in range(len(phis)):
            v = phis[r]
            if v in seen_vals:
                prev_idx = seen_vals[v]
                if prev_idx >= l:
                    l = prev_idx + 1
            
            seen_vals[v] = r
            
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start_idx = l
            
            # Early exit if we hit the cap
            if max_len >= 200:
                return 200, block_start + max_start_idx
        
        return max_len, block_start + max_start_idx

    # 4. Search Strategy
    # We know that long runs are more likely to be found in the higher ranges.
    # We will scan downwards from 10^12 in large blocks.
    # Block size: 100,000 seems reasonable. 
    # Time per block: ~0.5s - 1s. 
    # In 120s, we can scan ~100-200 blocks, covering 10^7 to 2*10^7 range.
    # This is a significant portion of the "interesting" high range.
    
    best_k = 1
    best_s = 1
    
    block_size = 100000
    current_end = MAX_S
    
    # We can also try some specific "hotspots" if time permits, 
    # but a dense scan of the top range is a good heuristic.
    
    while current_end > 1:
        elapsed = time.time() - start_time
        if elapsed > LIMIT - TIME_MARGIN:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        phis = segmented_phi(block_start, current_end)
        k, s = find_longest_distinct_run(phis, block_start)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k >= 200:
            return {'start': best_s, 'length': best_k}
        
        # Move to previous block
        current_end = block_start - 1
        
    # If we didn't find a very long run in the top 10^7, 
    # maybe we should try some random samples in the middle?
    # Or just return the best found.
    # Given the time, the top scan is the most robust.
    
    return {'start': best_s, 'length': best_k}
