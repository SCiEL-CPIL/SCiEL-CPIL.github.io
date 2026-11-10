"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

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
        # Vectorized: for each power p^k, multiply sf at indices divisible by p^k
        pk = p
        while pk <= max_val:
            # All multiples of pk get an additional factor of p
            # Use slice assignment for speed
            sf[pk::pk] *= p
            bitmask[pk::pk] |= bit
            pk *= p
        # Check time
        if time.time() > deadline - 15:
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

    # Sort candidates by rough count descending - windows with more roughs are more promising
    sorted_indices = np.argsort(-rough_counts)

    best_score = 25
    best_n = min_n

    # First pass: check top candidates by rough count
    # A window with r roughs has at most (24-r) + 1 distinct sf values
    # (all roughs contribute sf=1, non-roughs each contribute their own sf)
    # So if r >= 23, distinct count <= 2
    # If r >= 22, distinct count <= 3
    # etc.

    checked = 0
    max_check = 300000  # check up to 300k top candidates

    for idx in sorted_indices[:max_check]:
        if time.time() > deadline - 5:
            break

        n = int(n_arr[idx])
        r = int(rough_counts[idx])

        # Upper bound on distinct count: 1 (for all roughs) + (24-r) for non-roughs
        # But non-roughs might share sf values, so actual count <= this bound
        upper_bound = 1 + (24 - r)

        if upper_bound >= best_score:
            # Even the best case for this window can't beat current best
            # Since we're going in order of decreasing rough count, all subsequent
            # windows have <= roughs, so their upper bounds are >= this one
            # We can stop
            break

        # Count distinct small-prime factors in [n+1, n+24]
        window_sf = sf[n+1:n+25]
        count = len(set(window_sf.tolist()))

        if count < best_score:
            best_score = count
            best_n = n
            if best_score == 1:
                break

        checked += 1

    # If best_score is still high, do a broader search with vectorized processing
    if best_score > 3:
        # Process remaining windows in chunks
        chunk_size = 100000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline - 3:
                break

            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

            # For each n in chunk, we need sf[n+1:n+25]
            indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
            window_2d = sf[indices]  # shape (chunk_size, 24)

            # Count distinct values per row using sort + diff
            sorted_rows = np.sort(window_2d, axis=1)
            diffs = np.diff(sorted_rows, axis=1)
            distinct_counts = 1 + np.sum(diffs != 0, axis=1)

            # Find the minimum in this chunk
            min_idx = int(np.argmin(distinct_counts))
            min_count = int(distinct_counts[min_idx])

            if min_count < best_score:
                best_score = min_count
                best_n = int(chunk_n[min_idx])
                if best_score == 1:
                    break

    return best_n
# EVOLVE-BLOCK-END
