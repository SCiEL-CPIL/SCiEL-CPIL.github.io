import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    MAX_VAL = 1000001
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = [1] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        x = i
        while x > 1:
            p = spf[x]
            lpf[i] = max(lpf[i], p)
            x //= p
            while x > 1 and x % p == 0:
                x //= p
    
    # Vectorized scan
    lpf_arr = np.array(lpf)
    vals = np.maximum(lpf_arr[10000:1000001], lpf_arr[10001:1000002])
    best_idx = np.argmin(vals)
    best_n = 10000 + best_idx
    
    return best_n
