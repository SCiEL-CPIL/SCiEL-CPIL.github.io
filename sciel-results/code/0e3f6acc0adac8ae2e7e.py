"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Precompute the largest prime factor (LPF) for every integer up to
    1,000,001 using a sieve. Then scan all candidates n in [10000, 1000000]
    and track the minimum of max(lpf[n], lpf[n+1]).
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    N_MAX_PLUS_1 = N_MAX + 1
    
    # Precompute largest prime factor for all numbers up to N_MAX_PLUS_1
    lpf = [0] * (N_MAX_PLUS_1 + 1)
    
    # Sieve: for each prime p, set lpf[p] = p, and for multiples, update lpf
    for p in range(2, N_MAX_PLUS_1 + 1):
        if lpf[p] == 0:  # p is prime
            for multiple in range(p, N_MAX_PLUS_1 + 1, p):
                lpf[multiple] = p
    
    # Scan all candidates
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        val = lpf[n] if lpf[n] > lpf[n + 1] else lpf[n + 1]
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
