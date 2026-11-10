import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N = 1000000
    window = 24
    max_val = N + window  # need up to N+24
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute mask for each number: bit i set if prime[i] divides the number
    # Use sieve-style approach
    mask_arr = np.zeros(max_val + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        bit = 1 << i
        # Mark all multiples of p
        mask_arr[p::p] |= bit
    
    # Sliding window OR is not trivially subtractive, so we need a different approach
    # For each position, we need OR of masks[i] for i in [n+1, n+24]
    # Since OR is not invertible, use a different method:
    # For each prime, track if it appears in the window using a count
    
    # Alternative: for each n, check which primes appear in [n+1, n+24]
    # A prime p appears in [n+1, n+24] iff there exists k in [n+1, n+24] divisible by p
    # i.e., iff the range [n+1, n+24] contains a multiple of p
    # This happens iff floor((n+24)/p) > floor(n/p)
    # i.e., there's a multiple of p in (n, n+24]
    
    # So for each n, the set of primes that appear = {p : floor((n+24)/p) > floor(n/p)}
    # Count = number of such primes
    
    # We can vectorize this!
    # For n in [1000, 1000000], compute for each prime whether a multiple falls in (n, n+24]
    
    n_arr = np.arange(1000, N + 1, dtype=np.int64)  # 999001 values
    
    total_count = np.zeros(len(n_arr), dtype=np.int32)
    
    for p in primes:
        # A multiple of p is in (n, n+24] iff floor((n+24)/p) > floor(n/p)
        # Equivalent: (n+24) // p > n // p
        # Or: there exists integer k such that n < k*p <= n+24
        # i.e., ceil((n+1)/p) * p <= n+24
        # i.e., n+1 <= k*p <= n+24 for some k
        # The smallest multiple >= n+1 is: k = ceil((n+1)/p), multiple = k*p
        # It's in range iff k*p <= n+24
        
        # Vectorized: for each n, check if (n+1 + p - 1)//p * p <= n + 24
        # i.e., ((n + p) // p) * p <= n + 24
        # But this is per-element. Let me think differently.
        
        # A multiple of p exists in (n, n+24] iff:
        # (n+24) // p > n // p
        # Let's compute this vectorized
        left = (n_arr) // p       # floor(n/p)
        right = (n_arr + 24) // p  # floor((n+24)/p)
        has_multiple = (right > left).astype(np.int32)
        total_count += has_multiple
    
    # Find n with minimum total_count
    min_val = np.min(total_count)
    # Find all n with this minimum, pick the first one
    candidates = np.where(total_count == min_val)[0]
    if len(candidates) > 0:
        best_n = int(n_arr[candidates[0]])
    else:
        best_n = 1000
    
    # Verify we're within deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen but safety
        pass
    
    return best_n
# EVOLVE-BLOCK-END
