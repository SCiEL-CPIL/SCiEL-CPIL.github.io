import time
import math

def construct():
    # We need P(n) for n in [10000, 1000000] and P(n+1) for n in [10000, 999999]
    # So we need P up to 1000001
    limit = 1000001
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin before 90s limit
    
    # Sieve of Eratosthenes for Smallest Prime Factor
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts
    P = [1] * (limit + 1)
    for i in range(2, limit + 1):
        m = i
        res = 1
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        P[i] = res
    
    # Check deadline before search
    if time.time() > deadline:
        return 10000
    
    # Find max product
    best_n = 10000
    best_val = -1
    
    for n in range(10000, 1000000):
        if time.time() > deadline:
            break
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
