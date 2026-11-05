import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    L = 10000
    R = 1000000
    max_val = R + 1  # we need lpf up to 1000001
    
    # Sieve for Largest Prime Factor (LPF)
    # lpf[i] will hold the largest prime factor of i
    # Initialize all to 1
    lpf = [1] * (max_val + 1)
    
    # For each prime p, mark all multiples with p
    # Since we iterate p in increasing order, the last (largest) prime to divide m wins
    for p in range(2, max_val + 1):
        if lpf[p] == 1:  # p is prime
            for m in range(p, max_val + 1, p):
                lpf[m] = p
    
    # Now find n in [L, R] that minimizes max(lpf[n], lpf[n+1])
    best_n = L
    best_val = max(lpf[L], lpf[L + 1])
    
    # Check time periodically to avoid overhead
    check_interval = 50000
    for n in range(L + 1, R + 1):
        if (n - L) % check_interval == 0:
            if time.time() > deadline:
                break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
