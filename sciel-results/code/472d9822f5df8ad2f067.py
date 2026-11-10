import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    size = N_MAX + 2  # Need to check n+1 up to 1000001
    
    # Precompute largest prime factor for all numbers up to size
    lpf = np.zeros(size, dtype=np.int32)
    
    # Sieve: for each prime p, mark all multiples with lpf = p
    # We iterate p from 2 to size-1; if lpf[p] == 0, p is prime
    for p in range(2, size):
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p
    
    # Check time
    if time.time() > deadline:
        return N_MIN
    
    # Compute max(lpf[n], lpf[n+1]) for n in [N_MIN, N_MAX]
    # lpf[n] for n in [N_MIN, N_MAX] and lpf[n+1] for n+1 in [N_MIN+1, N_MAX+1]
    scores = np.maximum(lpf[N_MIN:N_MAX+1], lpf[N_MIN+1:N_MAX+2])
    
    # Find the minimum
    min_idx = np.argmin(scores)
    best_n = N_MIN + int(min_idx)
    
    return int(best_n)
