import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Sieve to compute largest prime factor for every number up to N
    lp = [0] * (N + 1)
    for i in range(2, N + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i  # i is the largest prime factor of j at this point
    
    # lp[i] now holds the largest prime factor of i
    # lp[1] = 0, but we define P(1) = 1
    if lp[1] == 0:
        lp[1] = 1
    
    # Scan all pairs a + b = N, find the one minimizing max(P(a), P(b))
    # Using local variables for speed
    lp_local = lp
    N_local = N
    best_max = N_local + 1
    best_a = 1
    best_b = N_local - 1
    
    for a in range(1, N_local // 2 + 1):
        b = N_local - a
        pa = lp_local[a]
        pb = lp_local[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            best_b = b
    
    return [best_a, best_b]
