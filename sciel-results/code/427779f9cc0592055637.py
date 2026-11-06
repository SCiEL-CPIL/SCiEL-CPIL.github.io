"""Sieve-based sliding window search for minimum distinct smooth factors."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 1000000
    W = 24  # window size

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)

    # Precompute smooth part for every number in [1, N]
    # smooth_part[m] = product of prime powers p^a for p < 24 where p^a || m
    smooth_part = [1] * (N + 1)
    for p in primes:
        for m in range(p, N + 1, p):
            v = m
            while v % p == 0:
                smooth_part[m] *= p
                v //= p

    # Precompute a compact encoding of the smooth part for each number
    # We need to count distinct smooth parts in a window of 24
    # Use the smooth_part values directly

    best_n = 1000
    best_count = None

    # Sliding window over positions 1..W-1 (so n+i for i=1..24, n in [1000, N-24])
    # Window for n: indices n+1, n+2, ..., n+24

    # Use a frequency dictionary for the current window
    from collections import defaultdict
    freq = defaultdict(int)

    # Initialize window for n = 1000: indices 1001..1024
    for i in range(1001, 1001 + W):
        sp = smooth_part[i]
        freq[sp] += 1

    distinct = len(freq)
    best_n = 1000
    best_count = distinct

    # Slide the window
    for n in range(1001, N - W + 1):
        # Remove index n (which was n-1+1 = n, the leftmost of previous window)
        # Previous window for n-1: indices n..n+W-1
        # Current window for n: indices n+1..n+W
        # So remove index n, add index n+W
        sp_remove = smooth_part[n]
        freq[sp_remove] -= 1
        if freq[sp_remove] == 0:
            del freq[sp_remove]

        sp_add = smooth_part[n + W]
        freq[sp_add] += 1

        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n

        # Check deadline periodically
        if n % 100000 == 0 and time.time() > deadline:
            break

    return best_n
