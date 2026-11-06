"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts for all numbers up to 1000024
    max_n = 1000000
    max_val = max_n + 25  # We need n+24 where n up to 1000000
    
    # smooth_part[m] = factor of m composed of primes < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            m = multiple
            while m % p == 0:
                smooth_part[multiple] *= p
                m //= p
    
    # Convert to Python list for faster access in tight loop
    sp = smooth_part.tolist()
    
    best_n = 1000
    best_score = 25  # worst case
    
    # Search for n in [1000, 1000000]
    n = 1000
    while n <= max_n:
        if time.time() > deadline:
            break
        
        # Count distinct in window of 24
        window = sp[n+1 : n+25]
        distinct = len(set(window))
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score <= 2:
                pass
        
        n += 1
    
    return best_n
# EVOLVE-BLOCK-END
