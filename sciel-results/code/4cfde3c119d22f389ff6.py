import math
import time

def construct():
    MAX_N = 1000001
    MIN_N = 10000
    
    # Build smallest prime factor sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts
    pp = [1] * (MAX_N + 1)
    for i in range(2, MAX_N + 1):
        n = i
        val = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[i] = val
        
    best_n = MIN_N
    best_product = 0
    
    # Find max product
    for n in range(MIN_N, 1000000):
        prod = pp[n] * pp[n+1]
        if prod > best_product:
            best_product = prod
            best_n = n
            
    return best_n
