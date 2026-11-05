import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # safety margin before 90s kill

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] using NumPy sieve
    E = np.ones(MAX_VAL + 1, dtype=np.int64)
    indices = np.arange(MAX_VAL + 1)

    for p in small_primes:
        power = p
        while power <= MAX_VAL:
            mask = indices % power == 0
            E[mask] *= p
            power *= p

    # Convert to Python list for faster indexing in the sliding window loop
    # Actually, numpy array indexing is fast enough. Let's use it directly.
    
    # Sliding window with array-based frequency counter
    # Values of E[m] are at most MAX_VAL, so we can use an array of size MAX_VAL+1
    freq = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    distinct_count = 0
    best_count = WINDOW + 1
    best_n = N_MIN

    # Initialize window for n = N_MIN: indices N_MIN+1 to N_MIN+WINDOW
    for i in range(N_MIN + 1, N_MIN + WINDOW + 1):
        val = E[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1

    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
        if best_count <= 2:
            return best_n

    # Slide window: for n from N_MIN+1 to N_MAX
    # Remove E[n], add E[n+WINDOW]
    E_local = E  # local reference for speed
    freq_local = freq
    WINDOW_local = WINDOW
    N_MAX_local = N_MAX
    start_time_local = start_time
    DEADLINE_local = DEADLINE
    time_local = time.time

    for n in range(N_MIN + 1, N_MAX_local + 1):
        if time_local() - start_time_local > DEADLINE_local:
            break

        val_out = E_local[n]
        freq_local[val_out] -= 1
        if freq_local[val_out] == 0:
            distinct_count -= 1

        val_in = E_local[n + WINDOW_local]
        if freq_local[val_in] == 0:
            distinct_count += 1
        freq_local[val_in] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 2:
                return best_n

    return best_n
