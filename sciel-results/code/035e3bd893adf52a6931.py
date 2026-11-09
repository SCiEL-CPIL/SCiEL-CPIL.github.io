"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N_MAX = 1000000
    N_MIN = 1000
    WINDOW_SIZE = 24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of numbers we need to factor: N_MIN+1 to N_MAX+24
    M_start = N_MIN + 1
    M_end = N_MAX + WINDOW_SIZE
    size = M_end - M_start + 1
    
    # S[j] will store the part of (M_start + j) composed of primes in small_primes
    S = np.ones(size, dtype=np.int64)
    
    # Precompute S using a sieve-like approach
    for p in small_primes:
        # Find the first index j such that M_start + j is divisible by p
        first_k = (M_start + p - 1) // p
        first_j = first_k * p - M_start
        
        for j in range(first_j, size, p):
            val = M_start + j
            # Extract all factors of p from val
            while val % p == 0:
                S[j] *= p
                val //= p

    # Sliding window optimization
    # We track the frequency of each distinct factor value in the current window
    freq = {}
    distinct_count = 0
    
    # Helper to add a value to the window
    def add_val(v):
        nonlocal distinct_count
        count = freq.get(v, 0)
        if count == 0:
            distinct_count += 1
        freq[v] = count + 1

    # Helper to remove a value from the window
    def remove_val(v):
        nonlocal distinct_count
        count = freq[v]
        if count == 1:
            del freq[v]
            distinct_count -= 1
        else:
            freq[v] = count - 1

    # Initialize the first window [0, WINDOW_SIZE)
    # This corresponds to n = N_MIN, covering numbers N_MIN+1 to N_MIN+24
    for k in range(WINDOW_SIZE):
        add_val(S[k])
        
    min_distinct = distinct_count
    best_n = N_MIN
    
    # Slide the window
    # Window i covers S[i : i+WINDOW_SIZE]
    # n = N_MIN + i
    # We slide from i=1 to i = N_MAX - N_MIN
    max_i = N_MAX - N_MIN
    for i in range(1, max_i + 1):
        # Remove the element leaving the window: S[i-1]
        remove_val(S[i-1])
        # Add the element entering the window: S[i+WINDOW_SIZE-1]
        add_val(S[i+WINDOW_SIZE-1])
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = N_MIN + i
            # If we find the theoretical minimum, we can stop early
            if min_distinct == 1:
                break

    return best_n

# EVOLVE-BLOCK-END
