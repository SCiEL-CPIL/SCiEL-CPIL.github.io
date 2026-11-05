import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    # Max value we need: 1000000 + 24 = 1000024
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        # Sieve-style: for each power p^k, multiply extracted by p at each multiple
        power = p
        while power <= MAX_VAL:
            extracted[power::power] *= p
            power *= p

    # Sliding window search using NumPy vectorization
    # For each n in [N_MIN, N_MAX], window = extracted[n+1 : n+WINDOW+1]
    # We want to minimize the number of distinct values in each window.

    # Build the full array of extracted values we need
    # Window for n uses indices n+1 .. n+WINDOW
    # So we need extracted[N_MIN+1 .. N_MAX+WINDOW]
    vals = extracted[N_MIN + 1 : N_MAX + WINDOW + 1]
    # vals[i] corresponds to extracted[N_MIN + 1 + i]
    # Window for n = N_MIN + j uses vals[j : j+WINDOW]

    n_windows = N_MAX - N_MIN + 1
    best_count = WINDOW + 1
    best_n = N_MIN

    # Vectorized approach: use a chunked sliding window with np.unique
    # Process in chunks to keep memory bounded
    CHUNK = 100000
    for chunk_start in range(0, n_windows, CHUNK):
        if time.time() - start_time > DEADLINE:
            break
        chunk_end = min(chunk_start + CHUNK, n_windows)
        # Number of windows in this chunk
        n_chunk = chunk_end - chunk_start
        # We need vals[chunk_start : chunk_start + n_chunk + WINDOW - 1]
        chunk_vals = vals[chunk_start : chunk_start + n_chunk + WINDOW - 1]

        # Compute distinct count for each sliding window
        # For each window of size WINDOW, count distinct values
        # Use a vectorized approach with np.apply_along_axis or a clever trick
        # 
        # Efficient approach: for each position, count how many times each value
        # appears in the window. This is hard to vectorize directly.
        # 
        # Alternative: use the fact that we just need the COUNT of distinct values.
        # We can use a running frequency approach but vectorized.
        #
        # Let's use a different strategy: for each chunk, compute the distinct
        # count using a loop but with optimized inner operations, or use
        # np.unique on reshaped windows (memory intensive).
        #
        # Given the constraint, let's use a hybrid: compute distinct counts
        # using a vectorized frequency-based method.
        
        # Method: for each distinct value v in chunk_vals, find all positions
        # where it appears, then for each window, count if v appears at least once.
        # This is O(unique * window_size) which can be large.
        
        # Simpler vectorized method using stride tricks:
        # Create a view of shape (n_chunk, WINDOW) where each row is a window
        if n_chunk * WINDOW <= 10**7:  # memory check: ~80MB for 10M int64
            windows = np.lib.stride_tricks.as_strided(
                chunk_vals,
                shape=(n_chunk, WINDOW),
                strides=(chunk_vals.strides[0], chunk_vals.strides[0])
            )
            # Count distinct values per window
            # This is still hard to vectorize for distinct count
            # Use np.apply_along_axis with np.unique
            # But that's slow. Let's try a different approach.
            
            # Alternative: compute for each window the number of distinct values
            # by checking if each element is the first occurrence in that window.
            # For element at position (i, j) in the window, it's the first occurrence
            # if there's no equal element at (i, j') for j' < j.
            # This is O(WINDOW^2) per window = O(n_chunk * WINDOW^2)
            
            # For WINDOW=24, WINDOW^2 = 576, and n_chunk up to 100000
            # That's 57.6M operations, which is feasible with NumPy.
            
            # For each pair (j1, j2) with j1 < j2, check if windows[:, j1] == windows[:, j2]
            # An element at position j2 is a "new" distinct value if it doesn't match
            # any element at positions j1 < j2 in the same window.
            
            # distinct_count[i] = sum over j of [windows[i,j] not in windows[i,0:j]]
            # = WINDOW - sum over j of [exists j' < j with windows[i,j'] == windows[i,j]]
            
            # Let's compute: for each j from 1 to WINDOW-1, 
            # match[j] = any(windows[:, j] == windows[:, j'] for j' < j)
            # distinct_count = WINDOW - sum(match[j] for j in 1..WINDOW-1)
            
            # But this is still O(WINDOW^2) with Python loops over j.
            # Let's vectorize over j pairs.
            
            # Actually, let's just use a simple but reasonably fast approach:
            # For each window, use np.unique. With 100k windows of size 24,
            # this should be fast enough in a loop.
            
            for i in range(n_chunk):
                if time.time() - start_time > DEADLINE:
                    break
                w = windows[i]
                # Count distinct
                # Use np.unique which sorts and removes duplicates
                cnt = len(np.unique(w))
                if cnt < best_count:
                    best_count = cnt
                    best_n = N_MIN + chunk_start + i
                    if best_count <= 2:
                        return best_n
        else:
            # Fallback: process in smaller chunks
            for i in range(n_chunk):
                if time.time() - start_time > DEADLINE:
                    break
                w = chunk_vals[i:i+WINDOW]
                cnt = len(np.unique(w))
                if cnt < best_count:
                    best_count = cnt
                    best_n = N_MIN + chunk_start + i
                    if best_count <= 2:
                        return best_n

    return best_n
