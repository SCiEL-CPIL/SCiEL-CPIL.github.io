import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin (90s kill)
    
    N = 123457
    
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    
    # Optimization: only iterate up to sqrt(N)
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if time.time() > deadline:
            break
        if spf[i] == i:  # i is prime
            # Mark multiples
            start = i * i
            for j in range(start, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor
    # lpf[1] = 1
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if time.time() > deadline:
            break
        p = spf[i]
        # lpf[i] is max of p and lpf of i/p
        lpf[i] = max(p, lpf[i // p])
        
    # Find optimal a, b
    best_a = 1
    best_b = N - 1
    best_val = max(lpf[1], lpf[N - 1])
    
    # Iterate a from 1 to N//2
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            
    return [best_a, best_b]
