"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Precompute largest prime factor for all numbers up to N_MAX + 1
    # using a sieve-like approach
    max_val = N_MAX + 1
    lpf = list(range(max_val + 1))  # lpf[i] = largest prime factor of i
    
    # Sieve: for each prime p, mark multiples
    for i in range(2, max_val + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i  # i is a prime factor of j, and since we go in increasing order,
                           # the last prime factor we hit is the largest
    
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    # Scan through the range
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() > deadline:
            break
        
        lp_n = lpf[n]
        if lp_n >= best_val:
            continue
        
        lp_np1 = lpf[n + 1]
        if lp_np1 >= best_val:
            continue
        
        candidate_val = lp_n if lp_n > lp_np1 else lp_np1
        if candidate_val < best_val:
            best_val = candidate_val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
