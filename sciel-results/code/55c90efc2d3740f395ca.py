"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    N2 = MAX_N + 2  # need index up to 1000001
    
    # Build smallest prime factor sieve up to N2-1
    spf = np.arange(N2, dtype=np.int32)
    
    # Sieve of Eratosthenes for SPF
    limit = int(math.isqrt(N2 - 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part(n) = product of p^e where e >= 2 in factorization
    # Use log-space to handle large products, but we need exact integer for comparison
    # Since the values can get large, let's compute them as integers but track in a way that's efficient
    
    # First pass: compute prime factorization info
    # For each n, we need to factor it using spf and extract the powerful part
    
    # We'll compute powerful parts one at a time but with early termination check
    pp = np.ones(MAX_N + 2, dtype=np.int64)
    
    last_computed = MIN_N - 1
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
        
        # Check time periodically
        if n % 10000 == 0 and time.time() > deadline:
            break
    
    # Find best n
    best_n = MIN_N
    best_val = 0
    
    # Only consider n where both pp[n] and pp[n+1] are computed
    end_range = min(MAX_N - 1, last_computed)
    
    for n in range(MIN_N, end_range + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
