"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part: for each m, the product of p^v_p for p in primes
    MAX_VAL = 1000024
    small_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Sieve-like: for each prime, compute its contribution
    for p in primes:
        # For each multiple of p, extract p^v
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                val //= p
            p_part = m // val  # p^v
            small_part[m] *= p_part
    
    # Now scan all windows [n+1, n+24] for n in [1000, 1000000]
    # and find the one with minimum number of distinct small_part values
    
    best_n = 1000
    best_distinct = 25  # max possible is 24
    
    # Use NumPy for efficient window scanning
    # small_part is 1-indexed, window for n is small_part[n+1:n+25]
    
    # We'll scan in chunks to allow early termination
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Get the 24 values
        vals = small_part[n+1:n+25]
        distinct = len(np.unique(vals))
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
