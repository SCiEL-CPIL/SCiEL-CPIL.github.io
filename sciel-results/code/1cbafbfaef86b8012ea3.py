import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    max_val = 1000000 + 24

    # Compute large_part[m] = m with all small-prime factors removed
    large_part = list(range(max_val + 1))
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            while large_part[multiple] % p == 0:
                large_part[multiple] //= p

    # smooth_part[m] = the part of m composed of primes < 24
    smooth_part = [m // large_part[m] for m in range(max_val + 1)]

    best_n = 1000
    best_count = float('inf')

    freq = defaultdict(int)
    distinct = 0

    # Initial window for n=1000: positions 1001 to 1024 inclusive
    for i in range(1001, 1025):
        sp = smooth_part[i]
        freq[sp] += 1
        if freq[sp] == 1:
            distinct += 1

    best_count = distinct
    best_n = 1000

    # Slide window: n from 1001 to 1000000
    for n in range(1001, 1000001):
        # Remove position n (was first element of previous window [n, n+23])
        sp_out = smooth_part[n]
        freq[sp_out] -= 1
        if freq[sp_out] == 0:
            del freq[sp_out]
            distinct -= 1

        # Add position n+24 (new last element of window [n+1, n+24])
        sp_in = smooth_part[n + 24]
        freq[sp_in] += 1
        if freq[sp_in] == 1:
            distinct += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n

        if time.time() > deadline:
            break

    return best_n
