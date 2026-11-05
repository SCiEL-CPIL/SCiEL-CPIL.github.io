import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Sieve for smallest prime factor up to N_MAX + 1
    spf = list(range(N_MAX + 2))
    for i in range(2, int(math.isqrt(N_MAX + 1)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        while n > 1:
            p = spf[n]
            lpf = p
            while n % p == 0:
                n //= p
        return lpf
    
    best_n = N_MIN
    best_val = float('inf')
    
    # Search all n in range
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        lpf_n = largest_prime_factor(n)
        lpf_np1 = largest_prime_factor(n + 1)
        val = max(lpf_n, lpf_np1)
        if val < best_val:
            best_val = val
            best_n = n
            if best_val <= 3:  # Very good, can't do much better
                break
    
    return best_n
