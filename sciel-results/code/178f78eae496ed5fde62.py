"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    N = max_val - lower + 1

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # masks[i] represents the small-prime part of (lower + i) as a bitmask
    # Bit j is set if primes[j] divides (lower + i)
    masks = [0] * N

    # Precompute masks
    for j, p in enumerate(primes):
        bit = 1 << j
        # Find first index i such that (lower + i) % p == 0
        rem = lower % p
        first_i = (p - rem) % p
        # Iterate through multiples
        for i in range(first_i, N, p):
            masks[i] |= bit

    # Sliding window to count distinct masks
    # There are 2^9 = 512 possible masks
    freq = [0] * 512
    distinct = 0
    best_n = lower
    best_count = 25

    # Initialize window for n = lower: covers numbers lower+1 to lower+24
    # Indices in masks: 1 to 24
    for i in range(1, 25):
        m = masks[i]
        if freq[m] == 0:
            distinct += 1
        freq[m] += 1

    if distinct < best_count:
        best_count = distinct
        best_n = lower

    # Slide the window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        # Window for n-1: indices [n-lower, n+23-lower]
        # Window for n:   indices [n+1-lower, n+24-lower]
        # Leaving index: n - lower
        # Entering index: n + 24 - lower
        
        out_idx = n - lower
        in_idx = n + 24 - lower

        m_out = masks[out_idx]
        freq[m_out] -= 1
        if freq[m_out] == 0:
            distinct -= 1

        m_in = masks[in_idx]
        if freq[m_in] == 0:
            distinct += 1
        freq[m_in] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            # If we reach 1, we can break, but 1 is impossible.
            # If we reach 2, it's very unlikely but possible?
            # We continue to ensure we find the global minimum.
            
    return best_n
# EVOLVE-BLOCK-END
