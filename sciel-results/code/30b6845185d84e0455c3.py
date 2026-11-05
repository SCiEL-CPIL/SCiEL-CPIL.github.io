import time
import numpy as np
from numpy.lib.stride_tricks import as_strided

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    # Using int64 to be safe, though int32 might suffice for max value < 10^6 * 23?
    # Max 23-smooth number < 10^6 is small.
    # 2^a 3^b 5^c ... < 10^6
    # 2^19 > 10^6, so max power of 2 is 18. 2^18 = 262144.
    # 2^18 * 3 = 786432.
    # 2^18 * 3 * 5 > 10^6.
    # So the max value of S(m) for m < 10^6 is well within int32.
    # But S(m) can be up to m itself. m < 10^6.
    # So int32 is safe.
    
    S = np.ones(MAX_VAL + 1, dtype=np.int32)
    
    # Vectorized sieve for small prime parts
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            # Multiply S at all indices that are multiples of p_power by p
            # This adds one factor of p to the count for each multiple of p
            # Repeating for p^2, p^3, etc. accumulates the correct power
            S[p_power::p_power] *= p
            p_power *= p

    # We want to find n in [N_MIN, N_MAX] minimizing distinct count in S[n+1 : n+1+WINDOW]
    total_windows = N_MAX - N_MIN + 1
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Process in chunks
    CHUNK_SIZE = 200000
    
    for chunk_start in range(0, total_windows, CHUNK_SIZE):
        if time.time() - start_time > DEADLINE:
            break
        
        chunk_end = min(chunk_start + CHUNK_SIZE, total_windows)
        n_start = N_MIN + chunk_start
        n_end = N_MIN + chunk_end  # exclusive
        
        # Windows start at n_start, ..., n_end-1
        # Each window i uses S[n_start + i + 1 : n_start + i + 1 + WINDOW]
        # So we need S[n_start+1 : n_end + WINDOW]
        arr_start = n_start + 1
        arr_end = n_end + WINDOW
        
        flat_arr = S[arr_start : arr_end]
        
        if flat_arr.size >= (chunk_end - chunk_start) + WINDOW - 1:
            row_stride = flat_arr.strides[0]
            chunk_len = chunk_end - chunk_start
            
            windows_2d = as_strided(
                flat_arr,
                shape=(chunk_len, WINDOW),
                strides=(row_stride, row_stride)
            )
            
            # Count distinct values per row
            # Sorting is O(W log W) per row. W=24.
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

    return best_n
