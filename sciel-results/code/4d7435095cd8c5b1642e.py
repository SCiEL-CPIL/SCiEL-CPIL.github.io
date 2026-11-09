"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute largest prime factor for all numbers up to N_MAX + 1
    # Using a sieve-like approach
    max_val = N_MAX + 1
    lp = [0] * (max_val + 1)  # lp[i] = largest prime factor of i
    
    for i in range(2, max_val + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                lp[j] = i  # i is the largest prime factor seen so far for j
    
    # Now find n in [N_MIN, N_MAX] that minimizes max(lp[n], lp[n+1])
    best_n = N_MIN
    best_score = max(lp[N_MIN], lp[N_MIN + 1])
    
    for n in range(N_MIN + 1, N_MAX + 1):
        score = max(lp[n], lp[n + 1])
        if score < best_score:
            best_score = score
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
