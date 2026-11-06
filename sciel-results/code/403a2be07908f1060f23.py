import time
import math

def construct():
    start_time = time.time()
    LIMIT = 1000000
    # We need to check up to LIMIT, and use LIMIT+1 for n+1
    N = LIMIT + 1
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute powerful part for each number
    # pp[i] will store the powerful part of i
    pp = [1] * (N + 1)
    pp[1] = 1
    
    # Optimization: Precompute powers?
    # Or just factorize each number.
    # 1M numbers, average number of prime factors is small (~log log N).
    # This should be fast enough.
    
    for i in range(2, N + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            e = 0
            # Calculate power of p in x
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                # Multiply by p^e
                # p**e can be precomputed or calculated
                res *= (p ** e)
        pp[i] = res
        
    # Check time elapsed
    if time.time() - start_time > 45:
        # If this part took too long, we might need a faster approach,
        # but SPF factorization is usually very fast in Python for 1M.
        pass

    # 3. Find n that maximizes pp[n] * pp[n+1]
    best_n = 10000
    max_val = -1
    
    for n in range(10000, LIMIT + 1):
        val = pp[n] * pp[n+1]
        if val > max_val:
            max_val = val
            best_n = n
            
    return best_n
