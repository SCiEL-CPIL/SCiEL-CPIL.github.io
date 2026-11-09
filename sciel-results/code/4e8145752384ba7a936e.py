"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 123457
    
    # Sieve for largest prime factor
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Find the minimum possible max prime factor by checking all pairs
    # but with early termination when we find a very good solution
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # First, do a quick scan to get a good initial bound
    # Check pairs where one number is small (likely smooth)
    for a in range(1, min(N // 2, 10000) + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
        if best_max_pf == 1:
            break
    
    # Now do a full scan with early termination based on current best
    for a in range(1, N // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            if best_max_pf == 2:
                # Can't do better than 2 (both would be powers of 2, but sum is odd)
                break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
