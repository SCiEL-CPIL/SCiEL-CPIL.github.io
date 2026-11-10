"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    SIZE = N_MAX + 1
    
    # Build smallest prime factor sieve
    spf = list(range(SIZE))
    for i in range(2, int(math.isqrt(SIZE)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, SIZE, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [N_MIN, N_MAX]
    pp = [1] * SIZE
    for m in range(N_MIN, N_MAX + 1):
        x = m
        result = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                result *= (p ** e)
        pp[m] = result
    
    # Find best consecutive pair
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
