"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    limit = MAX_N + 1
    
    # Direct Largest Prime Factor Sieve
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (limit + 1)
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    best_n = MIN_N
    best_val = float('inf')
    
    # Scan for the minimum of max(lpf(n), lpf(n+1))
    for n in range(MIN_N, MAX_N + 1):
        current_val = max(lpf[n], lpf[n+1])
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
