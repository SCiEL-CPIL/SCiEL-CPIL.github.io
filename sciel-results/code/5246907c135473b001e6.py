import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    MAX_VAL = 1000001  # n+1 max is 1000001
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = [0] * (MAX_VAL + 1)
    lpf[1] = 1
    for i in range(2, MAX_VAL + 1):
        x = i
        lp = 1
        while x > 1:
            p = spf[x]
            lp = p
            while x % p == 0:
                x //= p
        lpf[i] = lp
    
    # Scan for best n
    best_n = 10000
    best_val = max(lpf[10000], lpf[100001])
    
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        v = max(lpf[n], lpf[n+1])
        if v < best_val:
            best_val = v
            best_n = n
    
    return best_n
