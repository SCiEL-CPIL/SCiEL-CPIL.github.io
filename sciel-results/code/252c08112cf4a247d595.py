"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we return before the 90s kill
    # The algorithm is fast, so 85s is a huge safety margin
    deadline = start_time + 85.0
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # 1. Build Smallest Prime Factor (SPF) sieve
    # spf[i] will hold the smallest prime factor of i
    spf = list(range(MAX_N + 2))
    
    # Sieve of Eratosthenes variant to find SPF
    limit = int(math.isqrt(MAX_N + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number in [MIN_N, MAX_N]
    # powerful_part(n) = product of p^e for all prime factors p of n where exponent e >= 2
    pp = [1] * (MAX_N + 2)
    
    last_computed = MIN_N - 1
    # We iterate up to MAX_N to ensure pp[MAX_N] is computed, 
    # so that pp[n] and pp[n+1] are valid for n up to MAX_N-1
    for n in range(MIN_N, MAX_N + 1):
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
        last_computed = n
        
        # Time check: If we are running out of time, break and use what we have
        # This is unlikely to trigger given N=1e6, but it's a safety net
        if n % 10000 == 0 and time.time() > deadline - 5.0:
            break
    
    # 3. Find n that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = 0
    
    # We can only consider n where n+1 <= last_computed
    # So n <= last_computed - 1
    # Also n must be >= MIN_N and n+1 <= MAX_N (implied by last_computed logic if loop finished)
    end_range = min(MAX_N - 1, last_computed - 1)
    
    if end_range < MIN_N:
        return MIN_N

    for n in range(MIN_N, end_range + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
