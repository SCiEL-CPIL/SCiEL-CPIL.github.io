"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def construct():
    N_MAX = 1000000
    N_MIN = 1000
    WINDOW_SIZE = 24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    M_start = N_MIN + 1
    M_end = N_MAX + WINDOW_SIZE
    size = M_end - M_start + 1
    
    S = np.ones(size, dtype=np.int64)
    
    # Precompute S(m)
    for p in small_primes:
        # Find the first index j such that M_start + j is divisible by p
        # M_start + j = k * p  =>  j = k * p - M_start
        # We need j >= 0, so k * p >= M_start => k >= ceil(M_start / p)
        first_k = (M_start + p - 1) // p
        first_j = first_k * p - M_start
        
        # Iterate through all such j
        for j in range(first_j, size, p):
            val = M_start + j
            while val % p == 0:
                S[j] *= p
                val //= p

    # Sliding window with frequency count
    # Window covers S[i : i+WINDOW_SIZE]
    # i=0 corresponds to n=N_MIN.
    
    freq = defaultdict(int)
    distinct_count = 0
    
    # Inline add/remove for speed
    def add_val(v):
        nonlocal distinct_count
        if freq[v] == 0:
            distinct_count += 1
        freq[v] += 1
        
    def remove_val(v):
        nonlocal distinct_count
        freq[v] -= 1
        if freq[v] == 0:
            distinct_count -= 1

    # Fill initial window [0, WINDOW_SIZE)
    for k in range(WINDOW_SIZE):
        add_val(S[k])
        
    min_distinct = distinct_count
    best_n = N_MIN
    
    # Slide window
    # Next window starts at i=1, ends at i+WINDOW_SIZE
    # Remove S[0], Add S[WINDOW_SIZE]
    # ...
    # Remove S[i-1], Add S[i+WINDOW_SIZE-1]
    
    max_i = N_MAX - N_MIN
    for i in range(1, max_i + 1):
        remove_val(S[i-1])
        add_val(S[i+WINDOW_SIZE-1])
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = N_MIN + i
            
            if min_distinct == 1:
                break

    return best_n

# EVOLVE-BLOCK-END
