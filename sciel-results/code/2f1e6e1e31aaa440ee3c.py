"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 123457
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max_p = float('inf')
    best_a = 1
    
    for a in range(1, N):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_max_p:
            best_max_p = val
            best_a = a
            
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
