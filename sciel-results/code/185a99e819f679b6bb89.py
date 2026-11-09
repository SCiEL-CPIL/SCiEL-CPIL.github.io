import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    
    # Time budget guard
    DEADLINE = 85.0
    
    N_LIMIT = MAX_N + 1
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = list(range(N_LIMIT + 1))
    
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part using recurrence
    # P(n) = P(n/p) * p^e_part
    # If n = p^k * m, p not in m.
    # If k=1, P(n) = P(m) = P(n/p)
    # If k=2, P(n) = p^2 * P(m) = p^2 * P(n/p)  (since P(n/p)=P(p*m)=P(m))
    # If k>=3, P(n) = p^k * P(m) = p * (p^{k-1} * P(m)) = p * P(n/p)
    
    p_part = [1] * (N_LIMIT + 1)
    
    for i in range(2, N_LIMIT + 1):
        p = spf[i]
        k = i // p
        if k % p == 0:
            # p^2 divides i
            if (k // p) % p == 0:
                # p^3 divides i, so exponent in i is >= 3
                # P[i] = P[k] * p
                p_part[i] = p_part[k] * p
            else:
                # p^2 divides i, but p^3 does not, so exponent in i is 2
                # P[i] = P[k] * p^2
                p_part[i] = p_part[k] * (p * p)
        else:
            # p divides i, but p^2 does not, so exponent in i is 1
            # P[i] = P[k]
            p_part[i] = p_part[k]
            
    # 3. Search
    best_n = MIN_N
    max_score = 0
    
    # Convert to numpy array for faster access? 
    # Python list access is fast enough for 1M elements.
    # But multiplication in a loop is the main cost.
    
    for n in range(MIN_N, MAX_N + 1):
        score = p_part[n] * p_part[n+1]
        if score > max_score:
            max_score = score
            best_n = n
            
        if time.time() - start_time > DEADLINE:
            break
            
    return best_n
