import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 1000000
    best_n = 10000
    best_prod = 0
    
    # Sieve of Eratosthenes to find smallest prime factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all x in [1, N]
    pp = [1] * (N + 1)
    for x in range(2, N + 1):
        p = spf[x]
        y = x // p
        if y % p == 0:
            z = y // p
            if z % p == 0:
                # Exponent of p in x is >= 3
                # pp[x] = p^k * pp[m], pp[y] = p^(k-1) * pp[m]
                # pp[x] = pp[y] * p
                pp[x] = pp[y] * p
            else:
                # Exponent of p in x is 2
                # pp[x] = p^2 * pp[m], pp[y] = pp[m] (since exp in y is 1)
                # pp[x] = pp[y] * p^2
                pp[x] = pp[y] * p * p
        else:
            # Exponent of p in x is 1
            # pp[x] = pp[m], pp[y] = pp[m]
            # pp[x] = pp[y]
            pp[x] = pp[y]
    
    # Search for max product using precomputed array
    # We can use a simple loop, which is very fast now
    for n in range(10000, N):
        if time.time() > deadline:
            break
        prod = pp[n] * pp[n + 1]
        if prod > best_prod:
            best_prod = prod
            best_n = n
            
    return best_n
