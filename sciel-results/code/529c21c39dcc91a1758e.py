"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    N_MAX = 1000001  # We need lpf up to n+1 where n can be up to 1000000
    
    # Sieve to compute largest prime factor for each number
    lpf = [0] * (N_MAX + 1)
    for i in range(2, N_MAX + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N_MAX + 1, i):
                lpf[j] = i
    
    # Search for n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
