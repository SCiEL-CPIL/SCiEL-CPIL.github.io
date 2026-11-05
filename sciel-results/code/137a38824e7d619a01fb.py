import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    min_n = 10000
    
    # 1. Compute largest prime factor for all numbers up to N+1
    # We need lpf[n+1] for n=N, so we need lpf up to N+1
    max_val = N + 1
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
        if i % 100000 == 0 and time.time() > deadline:
            break

    # 2. Find n in [10000, N] minimizing max(lpf[n], lpf[n+1]) using NumPy
    lpf_arr = np.array(lpf, dtype=np.int32)
    # n ranges from min_n to N
    # lpf[n] is at index n
    # lpf[n+1] is at index n+1
    left = lpf_arr[min_n : N + 1]
    right = lpf_arr[min_n + 1 : N + 2]
    
    vals = np.maximum(left, right)
    best_idx = int(np.argmin(vals))
    best_n = min_n + best_idx
    
    return best_n
