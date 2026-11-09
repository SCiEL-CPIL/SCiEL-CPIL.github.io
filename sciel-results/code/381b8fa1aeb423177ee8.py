"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # We need powerful_part for n and n+1, so up to 1000001
    
    # Linear sieve to compute spf (smallest prime factor) and spf_power (exact power of spf dividing n)
    spf = [0] * (LIMIT + 1)
    spf_power = [1] * (LIMIT + 1)
    primes = []
    for i in range(2, LIMIT + 1):
        if spf[i] == 0:
            spf[i] = i
            spf_power[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > LIMIT:
                break
            spf[i * p] = p
            if p == spf[i]:
                spf_power[i * p] = spf_power[i] * p
                break
            else:
                spf_power[i * p] = p
    
    # Compute powerful part using DP
    pp = [1] * (LIMIT + 1)
    for n in range(2, LIMIT + 1):
        p = spf[n]
        pe = spf_power[n]  # pe = p^e where p^e || n
        if pe > p:  # exponent e >= 2
            pp[n] = pe * pp[n // pe]
        else:
            pp[n] = pp[n // p]
    
    # Scan for the best n in [N_MIN, N_MAX] maximizing pp[n] * pp[n+1]
    best_n = N_MIN
    best_product = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
