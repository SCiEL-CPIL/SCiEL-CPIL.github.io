import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    limit = 1000000
    max_val = limit + 1
    
    # Initialize largest prime factor array using NumPy for speed
    # We need indices up to max_val, so size is max_val + 1
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    
    # Sieve to find largest prime factor
    # We iterate p from 2 to max_val
    # If lpf[p] is 0, p is prime
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime, update all multiples of p
            # lpf[p::p] assigns p to all indices p, 2p, 3p, ...
            lpf[p::p] = p
        
        # Check time periodically to avoid timeout
        if p % 100000 == 0:
            if time.time() - start_time > deadline - 10:
                # Fallback if sieve takes too long (unlikely with NumPy)
                # But we need a valid n. 10000 is a safe default.
                return 10000

    # Check time after sieve
    if time.time() - start_time > deadline - 5:
        return 10000

    best_n = 10000
    min_score = float('inf')
    
    # Scan for the pair with minimum max LPF
    # We can vectorize this scan as well for extra speed
    # Create arrays for n and n+1
    # n ranges from 10000 to limit
    ns = np.arange(10000, limit + 1)
    # Get LPF for n and n+1
    lpf_n = lpf[ns]
    lpf_np1 = lpf[ns + 1]
    
    # Compute score for each n
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    min_score = scores[min_idx]
    best_n = ns[min_idx]
    
    return int(best_n)
