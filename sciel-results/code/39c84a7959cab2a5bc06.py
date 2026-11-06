"""Sieve-based sliding window search for minimum distinct smooth factors."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 1000000
    W = 24  # window size

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Precompute smooth part for every number in [1, N]
    # smooth_part[m] = product of prime powers p^a for p < 24 where p^a || m
    smooth_part = np.ones(N + 1, dtype=np.int64)
    for p in primes:
        for m in range(p, N + 1, p):
            v = m
            while v % p == 0:
                smooth_part[m] *= p
                v //= p

    best_n = 1000
    best_count = W  # worst case: all 24 are distinct

    # Use a frequency dictionary for the current window
    freq = {}

    # Initialize window for n = 1000: indices 1001..1024
    for i in range(1001, 1001 + W):
        sp = int(smooth_part[i])
        freq[sp] = freq.get(sp, 0) + 1

    distinct = len(freq)
    best_n = 1000
    best_count = distinct

    # Slide the window with periodic deadline checks
    check_interval = 50000
    next_check = check_interval

    for n in range(1001, N - W + 1):
        # Remove index n (leftmost of previous window)
        sp_remove = int(smooth_part[n])
        cnt = freq[sp_remove] - 1
        if cnt == 0:
            del freq[sp_remove]
        else:
            freq[sp_remove] = cnt

        # Add index n+W
        sp_add = int(smooth_part[n + W])
        freq[sp_add] = freq.get(sp_add, 0) + 1

        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 3:  # very good, can stop early
                break

        # Check deadline periodically
        if n >= next_check:
            if time.time() > deadline:
                break
            next_check += check_interval

    return best_n
