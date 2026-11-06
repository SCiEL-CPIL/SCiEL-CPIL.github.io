"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts for all numbers up to 1000024
    # smooth_part[m] = the factor of m composed of primes < 24
    max_val = 1000024
    smooth_part = [1] * (max_val + 1)
    
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            temp = m
            factor = 1
            while temp % p == 0:
                factor *= p
                temp //= p
            smooth_part[m] *= factor
    
    # Now scan for best n: minimize distinct smooth parts in window of 24
    best_n = 1000
    best_count = 25  # worst case is 24 distinct + possibly 1 more
    
    for n in range(1000, 1000000):
        if time.time() > deadline:
            break
        parts = set()
        for i in range(1, 25):
            parts.add(smooth_part[n + i])
        c = len(parts)
        if c < best_count:
            best_count = c
            best_n = n
            if best_count <= 2:
                break
    
    return best_n
# EVOLVE-BLOCK-END
