"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import numpy as np
from collections import defaultdict

def construct():
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # up to 1000024
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part[i] = product of p^e for all p in small_primes dividing i
    # Use numpy for speed
    small_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        # For each multiple, divide by p as many times as possible
        # We need to compute p^e for each multiple
        # Let's do this in a loop but vectorized per prime
        for multiple in multiples:
            val = multiple
            while val % p == 0:
                small_part[multiple] *= p
                val //= p
    
    # Convert to list for faster access in the sliding window
    small_part_list = small_part.tolist()
    
    # Sliding window to find n minimizing distinct small_part values in [n+1, n+24]
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: numbers 1001 to 1024
    freq = defaultdict(int)
    distinct = 0
    for i in range(1001, 1001 + WINDOW):
        sp = small_part_list[i]
        if freq[sp] == 0:
            distinct += 1
        freq[sp] += 1
    
    best_count = distinct
    best_n = 1000
    
    for n in range(1001, MAX_N + 1):
        # Remove n (which is (n-1)+1), add n+WINDOW
        old_val = small_part_list[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct -= 1
        
        new_val = small_part_list[n + WINDOW]
        if freq[new_val] == 0:
            distinct += 1
        freq[new_val] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
