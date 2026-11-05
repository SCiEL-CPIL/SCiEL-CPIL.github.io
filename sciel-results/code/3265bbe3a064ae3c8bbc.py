import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 120.0  # seconds
    MAX_S = 10**12
    TIME_MARGIN = 5.0 # Reserve time for final check

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
    # Returns a list of phis
    def segmented_phi(start, end):
        size = end - start + 1
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
            
            # Create indices of multiples of p in [start, end]
            multiples = np.arange(first_mult, end + 1, p)
            idxs = multiples - start
            
            # For each index, check if nums[idx] is divisible by p
            mask = nums[idxs] % p == 0
            
            if not np.any(mask):
                continue
            
            # Get the indices that need updating
            update_idxs = idxs[mask]
            
            # Remove all factors of p from nums[update_idxs]
            for idx in update_idxs:
                while nums[idx] % p == 0:
                    nums[idx] //= p
            
            # Update phi: multiply by (1 - 1/p) = (p-1)/p
            # phis[idx] -= phis[idx] // p
            phis[update_idxs] = phis[update_idxs] - phis[update_idxs] // p
        
        # Handle remaining prime factors > sqrt(end)
        # If nums[idx] > 1, it is a prime factor (or 1)
        remaining = nums > 1
        if np.any(remaining):
            rem_idxs = np.where(remaining)[0]
            for idx in rem_idxs:
                phis[idx] = phis[idx] - phis[idx] // nums[idx]
                
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
            
            if max_len >= 200:
                return 200, block_start + max_start_idx
        
        return max_len, block_start + max_start_idx

    # 4. Search Strategy
    # Phase 1: Targeted search using primorial-based windows
    # Numbers that are products of small primes have many factors,
    # leading to smaller phi values and potentially fewer collisions
    # in the range [n, n+k-1]
    
    best_k = 1
    best_s = 1
    
    # Generate small primorials for targeted search
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    primorials = []
    prod = 1
    for p in small_primes:
        prod *= p
        primorials.append(prod)
    
    # Phase 1a: Search around large primorial multiples
    # These numbers have many small prime factors, which may reduce
    # collision probability in consecutive ranges
    for prim in primorials:
        if time.time() - start_time > LIMIT - TIME_MARGIN:
            break
        
        # Search near MAX_S / prim * prim
        # This gives numbers around 10^12 / prim
        target_base = MAX_S // prim
        # Try several offsets around this base
        for offset_mult in range(10, -1, -1):
            if time.time() - start_time > LIMIT - TIME_MARGIN:
                break
            
            base = target_base * (offset_mult + 1)
            if base > MAX_S:
                continue
            
            # Search in a window of size 50000 centered around base
            block_start = max(1, base - 25000)
            block_end = min(MAX_S, base + 25000)
            
            phis = segmented_phi(block_start, block_end)
            k, s = find_longest_distinct_run(phis, block_start)
            
            if k > best_k:
                best_k = k
                best_s = s
            
            if best_k >= 200:
                return {'start': best_s, 'length': best_k}
    
    # Phase 1b: Random sampling of large numbers
    # Use deterministic pseudo-random offsets for reproducibility
    rng = random.Random(42)
    
    num_random_samples = 50
    for i in range(num_random_samples):
        if time.time() - start_time > LIMIT - TIME_MARGIN:
            break
        
        # Pick a random starting point in [10^11, 10^12]
        rand_start = rng.randint(10**11, 10**12 - 50000)
        block_end = min(MAX_S, rand_start + 50000)
        block_start = rand_start
        
        phis = segmented_phi(block_start, block_end)
        k, s = find_longest_distinct_run(phis, block_start)
        
        if k > best_k:
            best_k = k
            best_s = s
        
        if best_k >= 200:
            return {'start': best_s, 'length': best_k}
    
    # Phase 2: Exhaustive scan from MAX_S downwards
    block_size = 50000
    current_end = MAX_S
    
    while current_end > 1:
        if time.time() - start_time > LIMIT - TIME_MARGIN:
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
    
    return {'start': best_s, 'length': best_k}
