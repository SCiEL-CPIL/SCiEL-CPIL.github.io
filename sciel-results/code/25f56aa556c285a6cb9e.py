import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Generate smallest prime factor (SPF) sieve up to 1000001
    MAX_N = 1000001
    spf = list(range(MAX_N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute largest prime factor for all numbers in [10000, 1000001]
    # More efficient: for each prime, mark its multiples
    lpf_arr = [1] * (MAX_N + 1)
    for i in range(2, MAX_N + 1):
        if spf[i] == i:  # i is prime
            for j in range(i, MAX_N + 1, i):
                lpf_arr[j] = i  # since we iterate in increasing order, last prime factor is largest

    # Use numpy for fast scan
    lpf_np = np.array(lpf_arr[10000:MAX_N + 2], dtype=np.int32)
    # lpf_np[i] corresponds to lpf_arr[10000 + i]
    # We need max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # That's max(lpf_np[i], lpf_np[i+1]) for i in [0, 990000]
    vals = np.maximum(lpf_np[:-1], lpf_np[1:])
    idx = int(np.argmin(vals))
    best_n = 10000 + idx
    best_val = int(vals[idx])

    return best_n
