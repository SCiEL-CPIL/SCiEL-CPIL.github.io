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
    
    # Compute powerful part for each number using sieve-based approach
    # pp[n] = product of p^e for each prime p where p^e || n and e >= 2
    pp = [1] * (MAX_N + 2)
    
    # For each prime p, for each power p^e with e>=2, 
    # for each multiple m of p^e, if the exact exponent of p in m is e,
    # then multiply pp[m] by p^e
    # 
    # More efficiently: for each prime p, for each multiple m of p^2,
    # determine the exact exponent of p in m, and if >= 2, multiply by p^e
    
    # Alternative: use the spf array to factor each number
    # But let's use a sieve-based approach for speed
    
    # For each prime p:
    #   power = p^2
    #   while power <= MAX_N:
    #     for each multiple m of power:
    #       # m is divisible by p^2
    #       # Determine exact exponent of p in m
    #       x = m
    #       e = 0
    #       while x % p == 0:
    #         x //= p
    #         e += 1
    #       if e >= 2:
    #         pp[m] *= p**e
    #     power *= p
    # 
    # This is still slow. Let's use the direct factorization approach but optimized.
    
    # Direct factorization using spf
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
        
        # Check time every 50000 iterations to avoid overhead
        if n % 50000 == 0:
            if time.time() > deadline - 2:
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
