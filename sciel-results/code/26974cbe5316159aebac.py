"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    MAX_N = 1000000
    # We need lpf for numbers up to 1000001 because we check n+1 for n=1000000
    limit = MAX_N + 1
    
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (limit + 1)
    
    # Sieve to find largest prime factors
    # Iterating i from 2 to limit
    # If lpf[i] is 0, i is a prime
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    min_lpf_val = float('inf')
    best_n = 10000
    
    # Check all n in [10000, 1000000]
    for n in range(10000, MAX_N + 1):
        # Largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        # Using conditional expression for speed
        current_val = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
        if current_val < min_lpf_val:
            min_lpf_val = current_val
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
