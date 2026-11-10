"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(small_primes)

    # Max value we need to precompute: n+24 where n <= 1000000
    max_val = 1000024

    # Precompute small-prime factor for all numbers using vectorized sieve
    # sf[x] = product of prime factors p < 24 (with multiplicity)
    sf = np.ones(max_val + 1, dtype=np.int64)

    # Also track bitmask: which primes divide x (for quick window analysis)
    bitmask = np.zeros(max_val + 1, dtype=np.int32)

    for pi, p in enumerate(small_primes):
        bit = 1 << pi
        pk = p
        while pk <= max_val:
            # Use np.multiply with out to avoid temporary array creation
            np.multiply(sf[pk::pk], p, out=sf[pk::pk])
            bitmask[pk::pk] |= bit
            pk *= p
        if time.time() > deadline - 20:
            break

    # Now sf[x] gives the small-prime factor for each x
    # bitmask[x] tells which primes divide x

    window_size = 24
    min_n = 1000
    max_n = 1000000

    # Precompute cumulative sum of rough (bitmask==0) for quick filtering
    rough = (bitmask == 0).astype(np.int32)
    cum_rough = np.zeros(max_val + 2, dtype=np.int32)
    np.cumsum(rough, out=cum_rough[1:])

    # Compute rough counts for all valid n
    n_arr = np.arange(min_n, max_n + 1, dtype=np.int64)
    rough_counts = cum_rough[n_arr + 24] - cum_rough[n_arr]

    # Sort candidates by rough count descending
    sorted_indices = np.argsort(-rough_counts)

    best_score = 25
    best_n = min_n

    # Vectorized distinct count helper: for a 1D array of 24 values, count distincts
    def count_distinct_24(arr):
        """Count distinct values in a 24-element array using sort+diff."""
        s = np.sort(arr)
        return 1 + np.sum(s[1:] != s[:-1])

    # First pass: check top candidates by rough count
    max_check = 500000

    for idx in sorted_indices[:max_check]:
        if time.time() > deadline - 5:
            break

        n = int(n_arr[idx])
        r = int(rough_counts[idx])

        # Upper bound on distinct count
        upper_bound = 1 + (24 - r)

        if upper_bound >= best_score:
            break

        # Count distinct small-prime factors in [n+1, n+24]
        window_sf = sf[n+1:n+25]
        count = int(count_distinct_24(window_sf))

        if count < best_score:
            best_score = count
            best_n = n
            if best_score <= 2:
                # 2 is likely optimal; do a quick full scan to confirm
                break

    # If we found score 2, verify with a full vectorized scan
    if best_score <= 2:
        # Full vectorized scan to find if score 1 exists (unlikely) and confirm best
        chunk_size = 200000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline - 3:
                break

            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

            indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
            window_2d = sf[indices]

            sorted_rows = np.sort(window_2d, axis=1)
            diffs = np.diff(sorted_rows, axis=1)
            distinct_counts = 1 + np.sum(diffs != 0, axis=1)

            min_count = int(distinct_counts.min())
            if min_count < best_score:
                best_score = min_count
                best_n = int(chunk_n[distinct_counts.argmin()])
                if best_score == 1:
                    break

    # If best_score is still high, do a broader search with vectorized processing
    elif best_score > 3:
        chunk_size = 200000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline - 3:
                break

            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

            indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
            window_2d = sf[indices]

            sorted_rows = np.sort(window_2d, axis=1)
            diffs = np.diff(sorted_rows, axis=1)
            distinct_counts = 1 + np.sum(diffs != 0, axis=1)

            min_count = int(distinct_counts.min())
            if min_count < best_score:
                best_score = min_count
                best_n = int(chunk_n[distinct_counts.argmin()])
                if best_score <= 2:
                    break

    return best_n
# EVOLVE-BLOCK-END
