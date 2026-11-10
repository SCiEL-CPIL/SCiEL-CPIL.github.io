"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24

    # Efficient precompute: for each number, compute the product of p^v_p(n)
    # for all primes p < 24, using a sieve-style approach.
    # Use a flat array and process each prime's contribution.
    small_part = np.ones(max_val + 1, dtype=np.int64)
    for p in primes:
        # For each power p^k, multiply small_part at multiples of p^k by p
        p_power = p
        while p_power <= max_val:
            # Use slice assignment instead of arange for speed
            small_part[p_power::p_power] *= p
            p_power *= p

    if time.time() > deadline:
        return 1000

    best_n = 1000
    best_score = float('inf')

    # Sliding window over indices n+1 .. n+24 (24 elements).
    window_counter = Counter()
    for i in range(1, 25):
        window_counter[small_part[1000 + i]] += 1

    distinct_count = len(window_counter)
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
        if best_score == 1:
            return best_n

    n = 1000
    # Convert to list for faster indexing in the hot loop
    sp = small_part
    while n <= 999999:
        if time.time() > deadline:
            break

        n += 1
        remove_val = sp[n]
        add_val = sp[n + 24]

        c = window_counter[remove_val] - 1
        if c == 0:
            del window_counter[remove_val]
        else:
            window_counter[remove_val] = c

        window_counter[add_val] += 1

        distinct_count = len(window_counter)
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break

    return best_n
# EVOLVE-BLOCK-END
