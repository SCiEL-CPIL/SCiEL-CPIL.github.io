import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # we need LPF up to 1000001
    
    # Sieve to compute largest prime factor for each number up to MAX_VAL
    lpf = list(range(MAX_VAL + 1))  # lpf[i] will hold largest prime factor of i
    
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
    
    # Now scan [10000, 1000000] to find n minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, MAX_N + 1):
        if time.time() > deadline:
            break
        v = lpf[n] if lpf[n] >= lpf[n+1] else lpf[n+1]
        if v < best_val:
            best_val = v
            best_n = n
    
    return best_n
