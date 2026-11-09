import time

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # 1. Build SPF sieve
    spf = list(range(N_MAX + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # Helper to get largest prime factor
    def lpf(x):
        if x <= 1:
            return 1
        max_p = 1
        while x > 1:
            p = spf[x]
            max_p = max(max_p, p)
            while x % p == 0:
                x //= p
        return max_p
        
    best_n = N_MIN
    best_val = float('inf')
    
    # 2. Scan range
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
            
        v1 = lpf(n)
        v2 = lpf(n+1)
        current_max = v1 if v1 > v2 else v2
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
