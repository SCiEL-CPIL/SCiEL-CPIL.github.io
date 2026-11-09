import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 123457
    
    # Sieve for Smallest Prime Factor
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute Largest Prime Factor for each number
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            if p > max_pf:
                max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
        
    # Find the pair (a, b) minimizing max(P(a), P(b))
    best_val = N + 1
    best_a = 1
    best_b = N - 1
    
    # We can stop early if we find a pair with value 2 (both powers of 2), 
    # but N is odd, so one must be odd. 
    # Min possible value is the smallest prime factor of any number in the pair.
    # Actually, just scan all.
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            
    return [best_a, best_b]
