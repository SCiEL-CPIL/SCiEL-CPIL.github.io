"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Max value we need: n+24 where n <= 1000000
    max_val = 1000024

    # Precompute sf[x] = product of prime factors p < 24 (with multiplicity)
    # Using a more efficient sieve: for each prime, multiply into multiples
    sf = np.ones(max_val + 1, dtype=np.int64)

    for p in small_primes:
        pk = p
        while pk <= max_val:
            sf[pk::pk] *= p
            pk *= p
        if time.time() > deadline - 25:
            break

    # Map sf values to compact indices for faster distinct counting
    # Collect unique sf values
    unique_sf = np.unique(sf)
    sf_map = {int(v): i for i, v in enumerate(unique_sf)}
    num_unique = len(unique_sf)

    # Create a lookup: for each position, get the compact index
    sf_idx = np.array([sf_map[int(v)] for v in sf], dtype=np.int32)

    min_n = 1000
    max_n = 1000000

    best_score = 25
    best_n = min_n

    # Single comprehensive vectorized scan in chunks
    # For each chunk of n values, build the 24-window matrix and count distincts
    # using a histogram-based approach for speed

    chunk_size = 200000  # larger chunks for fewer iterations
    for chunk_start in range(min_n, max_n + 1, chunk_size):
        if time.time() > deadline - 10:
            break
        chunk_end = min(chunk_start + chunk_size, max_n + 1)
        chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

        # Build 2D array of sf indices for all windows
        offsets = np.arange(1, 25, dtype=np.int64)
        indices = chunk_n[:, None] + offsets[None, :]
        window_2d = sf_idx[indices]  # shape (num_chunks, 24)

        # Count distincts per row using a bitset approach
        # Since num_unique is manageable, use a boolean matrix
        # For each row, create a bitmask of which values appear
        # But num_unique could be large, so use sorting as fallback if needed

        # Alternative: use np.unique per row - but that's slow
        # Better: for small windows, sorting is fast
        sorted_rows = np.sort(window_2d, axis=1)
        diffs = np.diff(sorted_rows, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)

        min_count = int(distinct_counts.min())
        if min_count < best_score:
            best_score = min_count
            best_n = int(chunk_n[distinct_counts.argmin()])
            if best_score <= 2:
                break

    # If still > 3, do a finer-grained scan with smaller chunks for more precision
    if best_score > 3 and time.time() < deadline - 5:
        chunk_size = 50000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline - 3:
                break
            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

            offsets = np.arange(1, 25, dtype=np.int64)
            indices = chunk_n[:, None] + offsets[None, :]
            window_2d = sf_idx[indices]

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
