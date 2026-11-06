"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Upper bound for n is 1000000, so n+1 is 1000001
    N = 1000001
    
    # Initialize array for Largest Prime Factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    
    # Sieve to find Largest Prime Factor
    # We iterate i from 2 to N. If lpf[i] is 0, i is prime.
    # We then mark all multiples of i with i.
    # Since we iterate i in increasing order, the last prime i that divides j
    # will be the largest prime factor of j.
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_val = float('inf')
    
    # Search for n in [10000, 1000000]
    # minimizing max(lpf[n], lpf[n+1])
    for n in range(10000, 1000001):
        val_n = lpf[n]
        val_n1 = lpf[n + 1]
        current_max = val_n if val_n > val_n1 else val_n1
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
