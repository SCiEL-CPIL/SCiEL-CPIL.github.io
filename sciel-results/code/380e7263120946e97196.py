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

    # Fast vectorized precompute of the "small prime part" (product of p^k
    # for each p strictly below 24, over all numbers up to max_val).
    small_part = np.ones(max_val + 1, dtype=np.int64)
    for p in primes:
        p_power = p
        while p_power <= max_val:
            indices = np.arange(p_power, max_val + 1, p_power)
            small_part[indices] *= p
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
    while n <= 999999:
        if time.time() > deadline:
            break

        # Slide from window [n+1..n+24] to [n+2..n+25]:
        # remove index n+1, add index n+25.
        n += 1
        remove_val = small_part[n]        # index n (was n+1 for previous n)
        add_val = small_part[n + 24]      # index n+24 (new n+25)

        window_counter[remove_val] -= 1
        if window_counter[remove_val] == 0:
            del window_counter[remove_val]

        window_counter[add_val] += 1

        distinct_count = len(window_counter)
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break

    return best_n
# EVOLVE-BLOCK-END
