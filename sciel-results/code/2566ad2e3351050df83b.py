"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    spf = list(range(MAX_N + 2))
    spf[0] = 0
    spf[1] = 1
    
    # Sieve of Eratosthenes
    limit = int(math.isqrt(MAX_N + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number using direct factorization
    pp = [1] * (MAX_N + 2)
    
    # Direct factorization using spf - optimized
    for n in range(MIN_N, MAX_N + 2):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        
        # Check time every 100000 iterations to reduce overhead
        if n % 100000 == 0:
            if time.time() > deadline - 3:
                break
    
    # Find best n
    best_n = MIN_N
    best_val = 0
    
    for n in range(MIN_N, MAX_N):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
