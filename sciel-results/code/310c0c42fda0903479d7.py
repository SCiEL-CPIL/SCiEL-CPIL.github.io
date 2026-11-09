import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s buffer before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_N = N_MAX + 1  # we need n and n+1, so up to 1000001
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part[n] is the product of p^e for all prime factors p with exponent e >= 2
    powerful_part = [1] * (MAX_N + 1)
    
    for i in range(2, MAX_N + 1):
        n = i
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        powerful_part[i] = pp
        
    # Find the best n in [N_MIN, N_MAX] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = N_MIN
    best_val = powerful_part[N_MIN] * powerful_part[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
