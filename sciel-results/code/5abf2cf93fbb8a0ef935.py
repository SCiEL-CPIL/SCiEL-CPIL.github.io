"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using sieve-like approach
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    small_factor[0] = 0
    
    for p in small_primes:
        # For each multiple of p, multiply p into small_factor for each power
        # We can do this by iterating through multiples and dividing
        # Using NumPy for speed where possible, but the while loop is tricky
        # Stick to Python loop for precomputation as it's small (1M * 9)
        for multiple in range(p, max_val + 1, p):
            val = multiple
            while val % p == 0:
                small_factor[multiple] *= p
                val //= p
                
    # Convert to list for faster access in the scan loop if needed, 
    # or keep as numpy array. Numpy scalar access is slower than list.
    sf_list = small_factor.tolist()
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Iterate over valid n
    # Optimization: Check time periodically
    for n in range(1000, 1000001):
        if n % 10000 == 0 and time.time() > deadline:
            break
            
        # Extract factors for n+1 .. n+24
        # Using a set is efficient for small numbers
        factors = set()
        for i in range(1, 25):
            factors.add(sf_list[n + i])
        count = len(factors)
        if count < best_score:
            best_score = count
            best_n = n
            if best_score == 1:
                break  # can't do better than 1
                
    return best_n
# EVOLVE-BLOCK-END
