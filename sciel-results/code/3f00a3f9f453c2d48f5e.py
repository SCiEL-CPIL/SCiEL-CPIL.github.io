"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    MAX_VAL = N + WINDOW  # 1000024

    # Precompute small_part[m]: the largest divisor of m composed of primes in PRIMES
    small_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # For each prime p, multiply small_part[m] by p for every power p^k dividing m
    for p in PRIMES:
        power = p
        while power <= MAX_VAL:
            small_part[power::power] *= p
            power *= p
            if power > MAX_VAL:
                break

    # Now small_part[i] contains the required factor for i.
    # We need to find n in [1000, 1000000] minimizing distinct values in small_part[n+1 : n+25]
    
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0

    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1001, 1001 + WINDOW):
        v = small_part[i]
        freq[v] += 1
        if freq[v] == 1:
            distinct += 1

    best_n = 1000
    best_distinct = distinct

    # Slide the window
    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
        
        # Remove element at index n (which was in the previous window)
        v_remove = small_part[n]
        freq[v_remove] -= 1
        if freq[v_remove] == 0:
            distinct -= 1

        # Add element at index n + 24
        v_add = small_part[n + 24]
        freq[v_add] += 1
        if freq[v_add] == 1:
            distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
