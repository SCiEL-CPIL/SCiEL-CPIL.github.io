# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin from 90s limit
    
    target = 123457
    
    # Precompute largest prime factor (LPF) for all numbers up to target-1
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    if max_n >= 1:
        lpf[1] = 1
        
    # Sieve to find largest prime factor
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
                
    best_max = float('inf')
    best_pair = (1, target - 1)
    
    # Search for optimal a
    for a in range(1, target):
        if (a & 0xFF) == 0:
            if time.time() > deadline:
                break
                
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        
        if cur_max < best_max:
            best_max = cur_max
            best_pair = (a, b)
            if best_max <= 3:
                break
                
    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
