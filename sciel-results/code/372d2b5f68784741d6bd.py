import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 1000002  # We need n+1 up to 1000001, so size must be at least 1000002
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N))
    for i in range(2, int(math.isqrt(N)) + 1):
        if i * i >= N: break
        if spf[i] == i:  # i is prime
            for j in range(i*i, N, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute Largest Prime Factor (LPF) for each number
    lpf = [0] * N
    lpf[1] = 1
    for i in range(2, N):
        x = i
        last_prime = 1
        while x > 1:
            p = spf[x]
            last_prime = p
            while x % p == 0:
                x //= p
        lpf[i] = last_prime
        
    # 3. Scan for the minimum largest prime factor
    best_n = 10000
    best_val = float('inf')
    
    # Check time occasionally, though this loop is fast
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
