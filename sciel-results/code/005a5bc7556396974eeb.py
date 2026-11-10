import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    MAX_VAL = 1000001  # need n+1 up to 1000001
    
    # Build smallest prime factor (SPF) sieve
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers up to MAX_VAL
    lpf_arr = [1] * (MAX_VAL + 1)
    for x in range(2, MAX_VAL + 1):
        if time.time() > deadline:
            # Fallback: use what we have so far
            break
        last_prime = 1
        t = x
        while t > 1:
            p = spf[t]
            last_prime = p
            while t % p == 0:
                t //= p
        lpf_arr[x] = last_prime
    
    # Find n in [10000, 1000000] minimizing max(LPF(n), LPF(n+1))
    best_n = 10000
    best_val = max(lpf_arr[10000], lpf_arr[10001])
    
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        cur = lpf_arr[n] if lpf_arr[n] > lpf_arr[n + 1] else lpf_arr[n + 1]
        if cur < best_val:
            best_val = cur
            best_n = n
    
    return best_n
