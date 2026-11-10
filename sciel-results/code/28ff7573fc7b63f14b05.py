import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for evaluator overhead
    
    # Upper bound for values we need LPF for is 1000000 + 1 = 1000001
    max_val = 1000001
    
    # Use NumPy for faster LPF sieve
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve to find largest prime factors
    # For each prime p, assign it to all its multiples.
    # Since we iterate in increasing order, the last assignment (largest prime) wins.
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            lpf[p::p] = p
        
        # Time check periodically
        if p % 100000 == 0:
            if time.time() > deadline:
                return 10000
    
    # Time check
    if time.time() > deadline:
        return 10000

    best_n = 10000
    min_score = np.iinfo(np.int32).max
    
    # Scan for the n that minimizes max(lpf[n], lpf[n+1])
    # Vectorize the scan using NumPy
    n_range = np.arange(10000, 1000001)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    # Find the index with minimum score
    min_idx = np.argmin(scores)
    min_score = scores[min_idx]
    best_n = n_range[min_idx]
    
    # Early termination check: if we find a very low score
    if min_score <= 5:
        pass  # Already found, no need to do more
    
    return int(best_n)
