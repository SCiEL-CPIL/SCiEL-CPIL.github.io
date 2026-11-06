import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24

    # We need smooth parts for numbers from lower+1 to max_val
    start_num = lower + 1
    end_num = max_val
    count = end_num - start_num + 1  # 999024

    nums = np.arange(start_num, end_num + 1, dtype=np.int64)
    smooth = np.ones(count, dtype=np.int64)

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    for p in small_primes:
        temp = nums.copy()
        factor = np.ones(count, dtype=np.int64)
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            factor[mask] *= p
        smooth *= factor

    # Convert to Python ints list for faster dict operations in sliding window
    # Actually, let's keep as numpy and use int() conversion only when needed
    # Or better: convert to a regular list for speed in the hot loop
    smooth_list = smooth.tolist()

    # Sliding window: for each n in [lower, upper], look at smooth parts of n+1..n+24
    # Index mapping: number x corresponds to index x - start_num in smooth_list
    # For n, window covers numbers n+1 to n+24
    # Index range: (n+1 - start_num) to (n+24 - start_num)
    # n=lower=1000: indices 0 to 23
    # n=upper=1000000: indices (1000001-1001) to (1000024-1001) = 999000 to 999023

    best_n = lower
    best_count = 25  # Max possible distinct is 24

    freq = {}
    distinct = 0

    # Initialize window for n = lower: indices 0 to 23
    for i in range(24):
        v = smooth_list[i]
        if v not in freq:
            freq[v] = 1
            distinct += 1
        else:
            freq[v] += 1

    if distinct < best_count:
        best_count = distinct
        best_n = lower

    if best_count == 1:
        return best_n

    # Slide the window
    # For n going from lower+1 to upper:
    #   Previous window (for n-1): indices (n-1+1-start_num) to (n-1+24-start_num) = (n-start_num) to (n+23-start_num)
    #   Current window (for n): indices (n+1-start_num) to (n+24-start_num)
    #   Remove: index (n-start_num), i.e., number n
    #   Add: index (n+24-start_num), i.e., number n+24

    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        out_idx = n - start_num
        in_idx = n + 24 - start_num

        v_out = smooth_list[out_idx]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            distinct -= 1

        v_in = smooth_list[in_idx]
        if v_in not in freq:
            freq[v_in] = 1
            distinct += 1
        else:
            freq[v_in] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break

    return best_n
