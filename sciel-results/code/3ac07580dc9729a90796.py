"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

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
        p_power = p
        while p_power <= max_val:
            for m in range(p_power, max_val + 1, p_power):
                if m % (p_power * p) != 0:
                    smooth_part[m] *= p
            p_power *= p
    
    # Sliding window distinct count for efficiency
    best_n = 1000
    best_count = 25
    
    # Initialize window for n=1000: elements smooth_part[1001] to smooth_part[1024]
    freq = defaultdict(int)
    for i in range(1, 25):
        freq[smooth_part[1000 + i]] += 1
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Check time less frequently to reduce overhead
    check_interval = 50000
    last_check = start_time
    
    for n in range(1001, 1000000):
        if (n - 1001) % check_interval == 0:
            if time.time() > deadline:
                break
        
        # Remove smooth_part[n] (leftmost of previous window)
        old_val = smooth_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct -= 1
        
        # Add smooth_part[n+24] (new rightmost)
        new_val = smooth_part[n + 24]
        if new_val not in freq:
            distinct += 1
        freq[new_val] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 2:
                break
    
    return best_n
# EVOLVE-BLOCK-END
