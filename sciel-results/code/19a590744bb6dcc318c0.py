import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    N_MAX = 1000000
    N_MIN = 10000
    LIMIT = N_MAX + 1  # need up to 1000001
    
    # Build SPF sieve using list for speed
    spf = list(range(LIMIT + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers up to LIMIT
    lpf_arr = [1] * (LIMIT + 1)
    for i in range(2, LIMIT + 1):
        n = i
        lpf = 1
        while n > 1:
            p = spf[n]
            lpf = p
            while n % p == 0:
                n //= p
        lpf_arr[i] = lpf
    
    # Now scan: for each n, val = max(lpf_arr[n], lpf_arr[n+1])
    best_n = N_MIN
    best_val = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        val = lpf_arr[n] if lpf_arr[n] >= lpf_arr[n+1] else lpf_arr[n+1]
        if val < best_val:
            best_val = val
            best_n = n
            if best_val <= 2:  # Can't get better than 2
                break
    
    return best_n
