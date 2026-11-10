# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # need to factor up to 1000001
    
    # Build smallest prime factor sieve
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # Powerful part = product of p^e for each prime p with exponent e >= 2
    pp = [1] * (LIMIT + 1)
    
    for n in range(2, LIMIT + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
    
    # Find n in [MIN_N, MAX_N] maximizing pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
