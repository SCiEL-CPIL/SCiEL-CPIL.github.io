import math
import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    N = 1000000
    min_n = 10000
    
    # Sieve to compute largest prime factor for each number up to N+1
    lpf = [0] * (N + 2)
    for i in range(2, N + 2):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 2, i):
                lpf[j] = i  # i is a prime factor of j; since we go ascending, last one is largest
    
    # Find n in [min_n, N] minimizing max(lpf[n], lpf[n+1])
    best_n = min_n
    best_val = max(lpf[best_n], lpf[best_n + 1])
    
    for n in range(min_n + 1, N + 1):
        if time.time() > deadline:
            break
        v = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
        if v < best_val:
            best_val = v
            best_n = n
        if best_val == 2:
            break  # Can't do better than 2 (both n and n+1 powers of 2, impossible for n>=10000)
    
    return best_n
