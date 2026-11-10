import time
import math

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We aim to finish in < 5s.
    deadline = start_time + 80.0

    MAX_N = 1000000
    MIN_N = 10000
    
    # 1. Build Smallest Prime Factor (SPF) array up to MAX_N + 1
    limit = MAX_N + 1
    spf = list(range(limit + 1))
    
    # Sieve of Eratosthenes for SPF
    # Optimization: Only need to mark multiples starting from p*p
    sqrt_limit = int(math.isqrt(limit))
    for i in range(2, sqrt_limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each n in [1, limit]
    # pp[n] = product of p^e where p is prime, e >= 2, and p^e divides n
    pp = [1] * (limit + 1)
    
    # Factorization using precomputed SPF
    for n in range(2, limit + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 1
            x //= p
            while x > 1 and spf[x] == p:
                e += 1
                x //= p
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        
        # Time check
        if time.time() > deadline:
            # If we time out, return a safe default (though unlikely)
            return MIN_N

    # 3. Scan for n in [MIN_N, MAX_N] maximizing pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
            
        # Time check
        if time.time() > deadline:
            break

    return best_n
