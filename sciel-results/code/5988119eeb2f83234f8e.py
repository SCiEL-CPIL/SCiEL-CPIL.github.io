"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # we need n and n+1, so up to 1000001
    
    # powerful_part[i] will hold the powerful part of i
    powerful_part = [1] * (MAX_N + 1)
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each number m from 2 to MAX_N, factor it and compute its powerful part
    # powerful_part[m] = product of p^e for each prime power p^e | m with e >= 2
    
    for m in range(2, MAX_N + 1):
        x = m
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        powerful_part[m] = pp
    
    # Now find n in [10000, 1000000] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    # Check deadline
    if time.time() > deadline:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
