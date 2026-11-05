import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78  # Safety margin
    
    N = 1000000
    MIN_N = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    # Optimization: handle 2 separately or just let the loop do it
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part P(n)
    # P(n) is multiplicative.
    # P(p^k) = p^k if k >= 2 else 1.
    # Recurrence:
    # Let p = spf[n], m = n // p.
    # If m % (p*p) == 0: P[n] = P[m] * p
    # elif m % p == 0: P[n] = P[m] * p*p
    # else: P[n] = P[m]
    
    P = [1] * (N + 1)
    
    for i in range(2, N + 1):
        p = spf[i]
        m = i // p
        pp = p * p
        if m % pp == 0:
            P[i] = P[m] * p
        elif m % p == 0:
            P[i] = P[m] * pp
        else:
            P[i] = P[m]

    # 3. Find n in [MIN_N, N-1] that maximizes P[n] * P[n+1]
    best_n = MIN_N
    best_val = 0
    
    for n in range(MIN_N, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
