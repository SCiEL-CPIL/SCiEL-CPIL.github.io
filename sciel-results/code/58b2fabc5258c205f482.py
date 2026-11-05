import time

def construct():
    deadline = time.time() + 85.0  # 85s safety margin
    
    N = 1000000
    # Sieve for smallest prime factor up to N+1 (to handle n+1)
    limit = N + 1
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to get largest prime factor using SPF
    def lpf(x):
        if x <= 1:
            return 1
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = p  # p is prime, track max
            while x % p == 0:
                x //= p
        return max_pf
    
    best_n = 10000
    best_val = lpf(10000)
    if lpf(10001) > best_val:
        best_val = lpf(10001)
    
    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        ln = lpf(n)
        if ln > best_val:
            continue  # can't improve
        ln1 = lpf(n + 1)
        val = max(ln, ln1)
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
