import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    MAX_VAL = N_MAX + WINDOW

    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        power = p
        while power <= MAX_VAL:
            extracted[power::power] *= p
            power *= p

    vals = extracted[N_MIN + 1 : N_MAX + WINDOW + 1]

    n_windows = N_MAX - N_MIN + 1
    best_count = WINDOW + 1
    best_n = N_MIN

    CHUNK = 100000
    for chunk_start in range(0, n_windows, CHUNK):
        if time.time() - start_time > DEADLINE:
            break
        chunk_end = min(chunk_start + CHUNK, n_windows)
        n_chunk = chunk_end - chunk_start
        
        chunk_vals = vals[chunk_start : chunk_start + n_chunk + WINDOW - 1]
        
        windows = np.lib.stride_tricks.as_strided(
            chunk_vals,
            shape=(n_chunk, WINDOW),
            strides=(chunk_vals.strides[0], chunk_vals.strides[0])
        ).copy()
        
        # Vectorized distinct count via sorting
        # Sort each row
        windows_sorted = np.sort(windows, axis=1)
        # Count positions where element differs from previous
        # First element is always distinct
        distinct_counts = 1 + np.sum(windows_sorted[:, 1:] != windows_sorted[:, :-1], axis=1)
        
        min_idx = int(np.argmin(distinct_counts))
        min_val = int(distinct_counts[min_idx])
        
        if min_val < best_count:
            best_count = min_val
            best_n = N_MIN + chunk_start + min_idx
            if best_count <= 2:
                return best_n

    return best_n
