import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 1000001
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    
    # Sieve: for each prime, update all its multiples
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # Since we go in increasing order, last assignment is the largest prime factor
    
    # Now scan for the best n
    best_n = 10000
    best_val = max(lpf[10000], lpf[100001])  # placeholder, will be overwritten
    
    for n in range(10000, N):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
