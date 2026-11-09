"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 1000000
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 2)
    
    # Sieve of Eratosthenes variant to find largest prime factors
    # Iterate through potential primes
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is a prime
            for j in range(i, N + 2, i):
                lpf[j] = i
                
    min_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 999999] such that max(lpf[n], lpf[n+1]) is minimized
    # n+1 can be at most 1000000
    for n in range(10000, N):
        val = lpf[n]
        if lpf[n+1] > val:
            val = lpf[n+1]
            
        if val < min_val:
            min_val = val
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
