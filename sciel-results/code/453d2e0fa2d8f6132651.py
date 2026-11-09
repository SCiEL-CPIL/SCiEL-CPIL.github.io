"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW

    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # s[m] will store the part of m composed of primes < 24
    s = np.ones(MAX_VAL + 1, dtype=np.int64)

    # Sieve to compute s[m] using vectorized operations
    for p in PRIMES:
        # Find all multiples of p
        multiples = np.arange(p, MAX_VAL + 1, p)
        # For each multiple, compute the power of p dividing it
        for m in multiples:
            k = m
            while k % p == 0:
                s[m] *= p
                k //= p

    # Sliding window to find min distinct count
    freq = defaultdict(int)
    distinct = 0

    # Initialize first window for n=1000
    # s[1001] to s[1024]
    for i in range(1, WINDOW + 1):
        v = int(s[1000 + i])
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct += 1

    best_n = 1000
    best_distinct = distinct

    # Slide window
    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break

        # Remove s[n]
        v_out = int(s[n])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            distinct -= 1

        # Add s[n+24]
        v_in = int(s[n + 24])
        if v_in in freq:
            freq[v_in] += 1
        else:
            freq[v_in] = 1
            distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 1:
                return best_n

    return best_n

# EVOLVE-BLOCK-END
