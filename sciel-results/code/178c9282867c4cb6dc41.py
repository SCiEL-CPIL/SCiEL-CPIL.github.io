import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute smallest prime factor up to N_MAX + 1
    max_val = N_MAX + 1
    spf = np.arange(max_val + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Vectorized largest prime factor computation
    nums = np.arange(N_MIN, N_MAX + 2, dtype=np.int64)
    current = nums.copy()
    lpf = np.ones(len(current), dtype=np.int64)
    
    # Iteratively extract prime factors
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        mask = current > 1
        if not mask.any():
            break
        
        idx = np.where(mask)[0]
        p = spf[current[idx]]
        
        # Divide out all instances of p for each number
        vals = current[idx].copy()
        # Repeatedly divide by p while divisible
        for _ in range(20):  # max ~20 divisions for numbers up to 10^6
            divisible = vals % p == 0
            if not divisible.any():
                break
            vals[divisible] //= p
        
        current[idx] = vals
        lpf[idx] = p
    
    # For n = N_MIN + i, value is max(lpf[i], lpf[i+1])
    vals_to_consider = np.maximum(lpf[:-1], lpf[1:])
    
    best_idx = int(np.argmin(vals_to_consider))
    best_n = N_MIN + best_idx
    
    return best_n
