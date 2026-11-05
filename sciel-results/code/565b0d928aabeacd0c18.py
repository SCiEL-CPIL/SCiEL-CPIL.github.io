import time
import numpy as np
from numpy.lib.stride_tricks import as_strided

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # The window for n is extracted[n+1 : n+1+WINDOW]
    # We want distinct counts for n in [N_MIN, N_MAX]
    
    CHUNK_SIZE = 100000  # number of windows to process per chunk
    best_n = N_MIN
    best_count = WINDOW + 1
    
    total_windows = N_MAX - N_MIN + 1
    
    # Process in chunks using vectorized distinct counting
    for chunk_start in range(0, total_windows, CHUNK_SIZE):
        if time.time() - start_time > DEADLINE:
            break
        
        chunk_end = min(chunk_start + CHUNK_SIZE, total_windows)
        n_start = N_MIN + chunk_start
        n_end = N_MIN + chunk_end  # exclusive
        
        # Windows start at n_start, n_start+1, ..., n_end-1
        # Each window i uses extracted[n_start + i + 1 : n_start + i + 1 + WINDOW]
        arr_start = n_start + 1
        arr_end = n_end + WINDOW
        
        chunk_len = chunk_end - chunk_start
        flat_arr = extracted[arr_start : arr_end]
        
        if flat_arr.size >= chunk_len + WINDOW - 1:
            row_stride = flat_arr.strides[0]
            windows_2d = as_strided(
                flat_arr,
                shape=(chunk_len, WINDOW),
                strides=(row_stride, row_stride)
            )
            
            # Count distinct values per row via sorting
            sorted_windows = np.sort(windows_2d, axis=1)
            changes = np.diff(sorted_windows, axis=1)
            distinct_counts = 1 + np.sum(changes != 0, axis=1)
            
            local_min_idx = np.argmin(distinct_counts)
            local_min_val = int(distinct_counts[local_min_idx])
            
            if local_min_val < best_count:
                best_count = local_min_val
                best_n = n_start + int(local_min_idx)
                if best_count == 1:
                    break
        else:
            for j in range(chunk_len):
                n = n_start + j
                vals = extracted[n+1 : n+1+WINDOW]
                dc = len(set(vals.tolist()))
                if dc < best_count:
                    best_count = dc
                    best_n = n
                    if best_count == 1:
                        break

    return best_n
