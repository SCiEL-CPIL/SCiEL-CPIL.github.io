import math
import time

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
    
    # 2. Segmented Sieve for Phi
    # Computes phi for all n in [start, end]
    # Returns a list of phis
    def segmented_phi(start, end):
        size = end - start + 1
        # Initialize nums and phis
        # nums[i] holds the current value of the number (start+i) after removing prime factors
        # phis[i] holds the current value of phi(start+i)
        nums = [start + i for i in range(size)]
        phis = [start + i for i in range(size)]
        
        # Sieve with primes up to sqrt(end)
        # Note: We only need primes p such that p*p <= end
        # Actually, if p > sqrt(end), it can appear at most once.
        # The standard segmented sieve logic handles this.
        
        for p in primes:
            if p * p > end:
                break
                
            # Find the first multiple of p in [start, end]
            # first_mult = ceil(start / p) * p
            first_mult = ((start + p - 1) // p) * p
            
            # Iterate through multiples of p
            for multiple in range(first_mult, end + 1, p):
                idx = multiple - start
                # If nums[idx] is divisible by p, it means p is a factor
                # We must update phi
                if nums[idx] % p == 0:
                    # Remove all factors of p from nums[idx]
                    while nums[idx] % p == 0:
                        nums[idx] //= p
                    # Update phi: phi(n) = phi(n/p) * (p-1) if p divides n?
                    # Actually, standard formula: if p|n, phi(n) = phi(n/p^k) * p^(k-1)*(p-1)
                    # Easier: phi(n) = n * prod(1 - 1/p).
                    # We maintain phis[idx] as n * prod(1 - 1/q) for q processed.
                    # So we multiply by (1 - 1/p) = (p-1)/p.
                    # phis[idx] -= phis[idx] // p
                    phis[idx] -= phis[idx] // p
        
        # Handle remaining prime factors > sqrt(end)
        # If nums[idx] > 1, it is a prime factor (or 1)
        for i in range(size):
            if nums[i] > 1:
                phis[i] -= phis[i] // nums[i]
                
        return phis

    # 3. Search Strategy
    # Scan blocks from MAX_S downwards.
    # With segmented sieve, we can do larger blocks.
    # Let's try block size 50,000.
    # Time estimate: 50,000 numbers. Sieve steps sum 1/p ~ log log N.
    # Should be very fast, < 0.1s per block.
    # 120s / 0.1s = 1200 blocks * 50,000 = 60,000,000 range covered.
    # This is much better coverage than before.
    
    best_k = 1
    best_s = 1
    
    block_size = 50000
    current_end = MAX_S
    
    while current_end > 1:
        if time.time() - start_time > LIMIT - TIME_MARGIN:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        # Compute phis for this block
        phis = segmented_phi(block_start, current_end)
        
        # Find longest run of distinct values
        # Two-pointer technique
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
                best_k = 200
                best_s = block_start + max_start_idx
                return {'start': best_s, 'length': best_k}
        
        if max_len > best_k:
            best_k = max_len
            best_s = block_start + max_start_idx
            
        # Move to previous block
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
