"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    N = 123457
    
    # Precompute largest prime factor for all integers up to N using a sieve
    # lpf[n] = largest prime factor of n, with lpf[1] = 1
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Scan all splits a + b = N to minimize max(lpf[a], lpf[b])
    best_max_pf = float('inf')
    best_a = 1
    
    # Optimize: only need to check a from 1 to N//2 since (a,b) and (b,a) are symmetric
    # But we need the actual pair, so track best_a
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        m = lpf[a] if lpf[a] >= lpf[b] else lpf[b]
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
