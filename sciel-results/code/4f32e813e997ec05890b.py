import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Precompute smallest prime factor
    spf = list(range(N_MAX + 2))
    for i in range(2, int(math.isqrt(N_MAX + 1)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_MAX + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers
    lpf = [1] * (N_MAX + 2)
    for x in range(2, N_MAX + 2):
        max_pf = 1
        y = x
        while y > 1:
            p = spf[y]
            max_pf = p
            while y % p == 0:
                y //= p
        lpf[x] = max_pf
    
    best_n = N_MIN
    best_score = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        score = max(lpf[n], lpf[n + 1])
        if score < best_score:
            best_score = score
            best_n = n
    
    return best_n
