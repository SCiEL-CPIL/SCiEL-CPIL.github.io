def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    # Use a conservative deadline to ensure we return a valid result
    # The task allows 60s, we use 50s for safety.
    deadline = start_time + 50  
    
    N = 1000000
    limit = N + 1  # We need lpf up to N+1
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    # Only need to sieve up to sqrt(limit)
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Largest Prime Factor Sieve using NumPy
    # Initialize with 0. 1 has no prime factors, but n>=10000 so we don't care about 1.
    lpf = np.zeros(limit + 1, dtype=np.int32)
    # Iterate through primes in increasing order.
    # For each prime p, mark all multiples of p with p.
    # Since we go from small to large, the last write for a number k will be by its largest prime factor.
    for p in primes:
        lpf[p::p] = p
    
    # 3. Search for best n
    best_n = 10000
    best_score = np.iinfo(np.int32).max
    
    # Convert to lists for faster scalar access in the Python loop, 
    # or use vectorized ops if possible. 
    # Vectorized min/max over pairs is tricky with overlapping windows, 
    # but we can just compute the array of scores.
    
    # Vectorized approach for finding the minimum:
    # Create an array of scores for n in [10000, N]
    # score_n = max(lpf[n], lpf[n+1])
    
    # Extract the relevant slice
    # lpf is 0-indexed.
    # We need n from 10000 to 1000000.
    # lpf[n] and lpf[n+1]
    
    # Let's do it vectorized to be fast and safe.
    # n_range = np.arange(10000, N + 1)
    # scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    # min_idx = np.argmin(scores)
    # best_n = n_range[min_idx]
    # best_score = scores[min_idx]
    
    # This is O(N) memory for the slice, which is 10^6 integers. 4MB. Very safe.
    # And it's much faster than a Python loop.
    
    n_range = np.arange(10000, N + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    if len(scores) > 0:
        min_idx = np.argmin(scores)
        best_n = int(n_range[min_idx])
        best_score = int(scores[min_idx])
        
        # Optional: Check time, though vectorized ops are fast.
        if time.time() > deadline:
             pass # We already have the answer.
             
    return best_n
