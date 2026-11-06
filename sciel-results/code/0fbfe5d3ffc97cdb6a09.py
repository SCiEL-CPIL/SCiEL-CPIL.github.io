"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin vs 90s kill

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size

    nvals = max_val - lower + 1

    # ---- Step 1: NumPy sieve for the 23-smooth part of each integer ----
    # Use a more efficient approach: for each prime, divide and accumulate
    sp_part = np.ones(nvals, dtype=np.int64)
    # Work on a copy to track remaining factors
    remaining = np.arange(lower, max_val + 1, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        # Repeatedly divide by p, accumulating the p-factor in sp_part
        # This avoids the need to iterate over powers explicitly
        changed = True
        while changed:
            changed = False
            mask = (remaining % p == 0)
            if np.any(mask):
                changed = True
                sp_part[mask] *= p
                remaining[mask] //= p
    
    if time.time() > deadline:
        return lower

    # ---- Step 2: Map smooth-part values to compact integer IDs ----
    # Use a dictionary for potentially faster mapping if few unique values
    # But np.unique is efficient for large arrays
    uniq_vals, ids = np.unique(sp_part, return_inverse=True)
    n_unique = len(uniq_vals)
    ids = ids.astype(np.int32)

    if time.time() > deadline:
        return lower

    # ---- Step 3: Sliding window with O(1) frequency updates ----
    max_start = min(nvals - window_size, upper - lower)

    freq = np.zeros(n_unique, dtype=np.int32)
    distinct_count = 0

    # Initialize first window (i=0)
    for j in range(window_size):
        idx = int(ids[j])
        if freq[idx] == 0:
            distinct_count += 1
        freq[idx] += 1

    best_n = lower
    best_count = distinct_count

    # Slide the window
    for i in range(1, max_start + 1):
        out_idx = int(ids[i - 1])
        freq[out_idx] -= 1
        if freq[out_idx] == 0:
            distinct_count -= 1
        in_idx = int(ids[i + window_size - 1])
        if freq[in_idx] == 0:
            distinct_count += 1
        freq[in_idx] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = lower + i
            if best_count <= 2:
                break

        if i % 100000 == 0 and time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END
