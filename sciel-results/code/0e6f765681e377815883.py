import time
import math
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1  # 1000001
    
    # Sieve to find primes and compute largest prime factor for all numbers up to MAX_VAL
    # Using a modified sieve: for each prime p, mark all multiples with lpf = p
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve of Eratosthenes to find primes, and simultaneously compute lpf
    # For each prime p, for each multiple m of p, set lpf[m] = p
    # Since we go in increasing order of p, the final lpf[m] is the largest prime factor
    
    # We need primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
    
    # Now compute lpf: for each prime p, mark multiples
    for p in range(2, MAX_VAL + 1):
        if is_prime[p]:
            for m in range(p, MAX_VAL + 1, p):
                lpf[m] = p
    
    # Find n in [LO, HI] that minimizes max(lpf[n], lpf[n+1])
    best_n = LO
    best_val = max(lpf[LO], lpf[LO + 1])
    
    for n in range(LO, HI + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            if best_val == 2:
                break  # Can't do better than 2
    
    return best_n
