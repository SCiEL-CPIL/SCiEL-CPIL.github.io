import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    
    N = 123457
    
    # Compute smallest prime factor using sieve
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    # Linear scan for optimal pair: guaranteed global optimum in O(N)
    best_val = max(lpf[1], lpf[N - 1])
    best_a = 1
    for a in range(2, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    best_b = N - best_a
    return [best_a, best_b]
