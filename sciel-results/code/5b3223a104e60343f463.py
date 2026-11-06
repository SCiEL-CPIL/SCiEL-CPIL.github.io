import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    LIMIT = 1000000
    N = LIMIT + 1  # we need pp up to N (for n+1 when n=LIMIT)
    
    # 1. Sieve for Smallest Prime Factor (SPF) using NumPy for speed
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            # Mark multiples of i
            start = i * i
            step = i
            # Vectorized update: only update if spf[j] is still j (i.e., not yet marked by smaller prime)
            # This is tricky to do fully vectorized with conditionals, but we can do it in chunks
            # Actually, for SPF, the standard approach is:
            # for j in range(i*i, N+1, i):
            #     if spf[j] == j:
            #         spf[j] = i
            # Vectorizing this conditional is hard. Let's use a different approach.
            
            # Alternative: just loop, but it's fast enough for N=1M in Python?
            # Let's try a hybrid: use the loop but optimize inner.
            pass
    
    # Actually, let's do the SPF sieve in pure Python but optimized, or use a different method.
    # For N=1M, the SPF sieve in Python takes ~0.5-1s. Let's do it.
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() - start_time > DEADLINE - 30:
        # Not enough time, but this should be fast
        pass
    
    # 2. Compute powerful part for each number
    # pp[i] = product of p^e where e>=2 in factorization of i
    pp = [1] * (N + 1)
    
    # Optimization: Precompute p^e for common small primes?
    # Or just factorize each number using spf.
    
    for i in range(2, N + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        pp[i] = res
        
    if time.time() - start_time > DEADLINE - 10:
        # Time check
        pass
    
    # 3. Find n that maximizes pp[n] * pp[n+1]
    # Use NumPy for vectorized multiplication and max
    pp_arr = np.array(pp[10000:N+1], dtype=np.int64)
    # pp_arr[i] corresponds to pp[10000+i]
    # We want pp[n] * pp[n+1] for n in [10000, LIMIT]
    # That is pp_arr[i] * pp_arr[i+1] for i in [0, LIMIT-10000]
    
    # But pp_arr has length LIMIT - 10000 + 1 = 990001
    # We need products of adjacent elements: pp_arr[0]*pp_arr[1], pp_arr[1]*pp_arr[2], etc.
    # That gives us LIMIT - 10000 products.
    
    if len(pp_arr) < 2:
        return 10000
    
    products = pp_arr[:-1] * pp_arr[1:]
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)
