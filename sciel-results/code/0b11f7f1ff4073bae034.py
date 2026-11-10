import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need lpf up to N+1 = 1000001
    
    # Sieve for smallest prime factor
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number using SPF
    lpf = [0] * (max_val + 1)
    lpf[1] = 1
    for i in range(2, max_val + 1):
        x = i
        largest = 1
        while x > 1:
            p = spf[x]
            largest = p
            while x % p == 0:
                x //= p
        lpf[i] = largest
    
    # Find n in [10000, N] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, N + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
