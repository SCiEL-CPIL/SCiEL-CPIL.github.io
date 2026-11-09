"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor array
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each n
    powerful_part = [1] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        x = n
        pp = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count >= 2:
                pp *= p ** count
        powerful_part[n] = pp
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return a reasonable value
        return 10000
    
    # Find n in [MIN_N, MAX_N-1] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = MIN_N
    best_val = powerful_part[MIN_N] * powerful_part[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
