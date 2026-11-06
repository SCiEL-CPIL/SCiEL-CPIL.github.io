import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 80

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime parts
    arr = np.arange(lower, max_val + 1, dtype=np.int64)
    smooth_parts = np.ones(nvals, dtype=np.int64)

    for p in _SMALL_PRIMES:
        count = np.zeros(nvals, dtype=np.int32)
        t = arr.copy()
        while True:
            mask = (t % p == 0)
            if not np.any(mask):
                break
            t[mask] //= p
            count[mask] += 1

        if np.any(count > 0):
            powers = np.ones(nvals, dtype=np.int64)
            nz = count > 0
            powers[nz] = p ** count[nz]
            smooth_parts *= powers

    # Map smooth parts to integer IDs for fast array-based frequency tracking
    unique_sp, sp_ids = np.unique(smooth_parts, return_inverse=True)
    num_unique = len(unique_sp)
    sp_ids = sp_ids.astype(np.int32)

    best_n = lower
    best_count = 999

    # Frequency array indexed by sp_id
    freq_arr = np.zeros(num_unique, dtype=np.int32)
    distinct_count = 0

    # Initialize window for n = lower (indices 1 to 24)
    for i in range(1, 25):
        sid = sp_ids[i]
        if freq_arr[sid] == 0:
            distinct_count += 1
        freq_arr[sid] += 1

    if distinct_count < best_count:
        best_count = distinct_count
        best_n = lower
        if best_count == 1:
            return best_n

    # Slide window
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break

        sid_out = sp_ids[offset]
        sid_in = sp_ids[offset + 24]

        # Remove sid_out
        if freq_arr[sid_out] == 1:
            freq_arr[sid_out] = 0
            distinct_count -= 1
        else:
            freq_arr[sid_out] -= 1

        # Add sid_in
        if freq_arr[sid_in] == 0:
            freq_arr[sid_in] = 1
            distinct_count += 1
        else:
            freq_arr[sid_in] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
