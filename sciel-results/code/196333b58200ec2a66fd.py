import time
import numpy as np
from numpy.lib.stride_tricks import as_strided


def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    N = 1000000
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW  # 1000024

    # Compute sp[m] = product of p^v_p(m) for all primes p < 24, for m in 1..MAX_VAL
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in PRIMES:
        # All multiples of p
        multiples = np.arange(p, MAX_VAL + 1, p)
        # For each multiple, compute v_p(m) = exponent of p in m
        temp = multiples.copy()
        exp = np.zeros(len(multiples), dtype=np.int64)
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            exp[mask] += 1
            temp[mask] //= p
        # Multiply sp[multiples] by p^exp
        sp[multiples] *= np.power(p, exp, dtype=np.int64)

    # Now scan for n in [1000, N] minimizing distinct count of sp[n+1..n+24]
    best_n = 1000
    best_distinct = WINDOW + 1  # worst case: all 24 distinct

    # Pass 1: Coarse scan with step 5 to quickly find a good region
    step1 = 5
    for n in range(1000, N + 1, step1):
        if time.time() > deadline - 30:
            break
        window = sp[n + 1 : n + 1 + WINDOW]
        # Count distinct using sort for small window
        sorted_w = np.sort(window)
        diffs = np.diff(sorted_w)
        dc = int((diffs != 0).sum()) + 1
        if dc < best_distinct:
            best_distinct = dc
            best_n = n
            if best_distinct <= 2:
                break

    # Pass 2: Fine scan around best_n with step 1
    if time.time() < deadline - 15:
        ref_start = max(1000, best_n - 5000)
        ref_end = min(N, best_n + 5000)
        for n in range(ref_start, ref_end + 1):
            if time.time() > deadline - 10:
                break
            window = sp[n + 1 : n + 1 + WINDOW]
            sorted_w = np.sort(window)
            diffs = np.diff(sorted_w)
            dc = int((diffs != 0).sum()) + 1
            if dc < best_distinct:
                best_distinct = dc
                best_n = n
                if best_distinct <= 1:
                    break

    # Pass 3: Vectorized scan of remaining regions if time permits
    if time.time() < deadline - 10 and best_distinct > 2:
        chunk_size = 100000
        for chunk_start in range(1000, N + 1, chunk_size):
            if time.time() > deadline - 5:
                break

            # Skip regions already covered by pass 2
            if abs(chunk_start - best_n) < 6000:
                continue

            chunk_end = min(chunk_start + chunk_size, N + 1)
            num_windows = chunk_end - chunk_start

            # base array: sp[chunk_start+1 : chunk_start + num_windows + WINDOW]
            base = sp[chunk_start + 1 : chunk_start + num_windows + WINDOW]
            row_stride = base.strides[0]
            mat = as_strided(base, shape=(num_windows, WINDOW), strides=(row_stride, row_stride))

            # Sort each row and count distinct via diffs
            sorted_mat = np.sort(mat, axis=1)
            diffs = np.diff(sorted_mat, axis=1)
            distinct_counts = (diffs != 0).sum(axis=1) + 1

            local_min_idx = int(np.argmin(distinct_counts))
            local_min_val = int(distinct_counts[local_min_idx])

            if local_min_val < best_distinct:
                best_distinct = local_min_val
                best_n = chunk_start + local_min_idx
                if best_distinct <= 1:
                    break

    # Pass 4: Final refinement around best_n with step 1
    if time.time() < deadline - 3 and best_distinct > 1:
        ref_start = max(1000, best_n - 2000)
        ref_end = min(N, best_n + 2000)
        for n in range(ref_start, ref_end + 1):
            if time.time() > deadline - 1:
                break
            window = sp[n + 1 : n + 1 + WINDOW]
            sorted_w = np.sort(window)
            diffs = np.diff(sorted_w)
            dc = int((diffs != 0).sum()) + 1
            if dc < best_distinct:
                best_distinct = dc
                best_n = n

    return best_n
