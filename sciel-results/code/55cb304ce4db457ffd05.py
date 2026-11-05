# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Finds n in [10000, 1000000] that maximizes powerful_part(n) * powerful_part(n+1).
    powerful_part(k) is the product of p^e for all prime factors p of k where e >= 2.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 1000000
    START = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number
    # P[k] stores the powerful part of k
    P = [1] * (N + 1)
    
    for k in range(2, N + 1):
        x = k
        p_part = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part *= (p ** e)
        P[k] = p_part
    
    # 3. Find n in [10000, N-1] maximizing P[n] * P[n+1]
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
# EVOLVE-BLOCK-END
