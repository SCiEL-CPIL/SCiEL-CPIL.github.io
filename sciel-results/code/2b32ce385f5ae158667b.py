"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1
    
    # 1. Sieve of Eratosthenes to find all primes up to limit
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = b'\x00' * ((limit - start) // step + 1)
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # 2. Compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    lpf = [0] * (limit + 1)
    for p in primes:
        for multiple in range(p, limit + 1, p):
            lpf[multiple] = p
    
    # 3. For each n in [N_MIN, N_MAX], compute max(lpf[n], lpf[n+1])
    #    and find the n that minimizes this value.
    best_n = N_MIN
    best_val = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 2:
            break
        v1 = lpf[n]
        v2 = lpf[n + 1]
        v = v1 if v1 > v2 else v2
        if v < best_val:
            best_val = v
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
