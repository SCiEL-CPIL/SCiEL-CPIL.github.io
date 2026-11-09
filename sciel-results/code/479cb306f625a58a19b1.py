import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    
    # Time budget guard - keep safety margin before 90s kill
    DEADLINE = 85.0  # Seconds
    
    N_LIMIT = MAX_N + 1
    
    # 1. Compute Smallest Prime Factor (SPF) for all numbers up to N_LIMIT
    spf = list(range(N_LIMIT + 1))
    
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number using efficient factorization
    p_part = [1] * (N_LIMIT + 1)
    
    for i in range(2, N_LIMIT + 1):
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
        p_part[i] = res
    
    # 3. Identify candidate n where both P(n) > 1 and P(n+1) > 1
    # This reduces the search space significantly
    candidates = []
    for n in range(MIN_N, MAX_N + 1):
        if p_part[n] > 1 and p_part[n+1] > 1:
            candidates.append(n)
    
    # 4. Search for max product among candidates
    best_n = MIN_N
    max_score = 0
    
    # Check time before expensive search
    if time.time() - start_time > DEADLINE:
        return best_n
    
    for n in candidates:
        score = p_part[n] * p_part[n+1]
        if score > max_score:
            max_score = score
            best_n = n
    
    return best_n
