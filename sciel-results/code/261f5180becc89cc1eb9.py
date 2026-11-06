"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1
    
    # 1. Sieve to compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    lpf = [0] * (limit + 1)
    
    # We iterate i from 2 to limit.
    # If lpf[i] is 0, i is prime.
    # For each prime i, we update lpf for all its multiples.
    # Since we iterate i in increasing order, the last prime that updates a multiple
    # will be the largest prime factor of that multiple.
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
    
    # 2. Find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_score = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        # Check deadline periodically to be safe, though this loop is fast
        if time.time() > deadline:
            break
        
        # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        # because n and n+1 are coprime.
        current_score = max(lpf[n], lpf[n+1])
        
        if current_score < best_score:
            best_score = current_score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
