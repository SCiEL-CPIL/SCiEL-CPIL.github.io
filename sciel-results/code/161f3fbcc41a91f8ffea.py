import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000000
    START = 10000
    
    # Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number using SPF
    # P[k] = product of p^e for prime factors p with exponent e >= 2
    # Use recurrence: if spf[k] = p with exponent e:
    #   if e >= 2: P[k] = p^e * P[k // p^e]
    #   if e == 1: P[k] = P[k // p]
    P = [1] * (N + 1)
    
    for k in range(2, N + 1):
        p = spf[k]
        x = k
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        if e >= 2:
            P[k] = (p ** e) * P[x]
        else:
            P[k] = P[x]
    
    # Find n in [10000, N-1] maximizing P[n] * P[n+1]
    best_n = START
    best_val = P[START] * P[START + 1]
    
    for n in range(START, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
