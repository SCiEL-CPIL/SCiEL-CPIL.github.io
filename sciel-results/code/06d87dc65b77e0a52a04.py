"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # need powerful_part for n and n+1, so up to 1000001
    
    # Linear sieve to compute smallest prime factor (spf) and exponent of spf
    # spf[i] = smallest prime factor of i
    # exp_spf[i] = exponent of spf[i] in i's factorization
    # powerful[i] = powerful part of i
    
    spf = [0] * (LIMIT + 1)
    exp_spf = [0] * (LIMIT + 1)
    powerful = [1] * (LIMIT + 1)
    
    primes = []
    spf[0] = 0
    spf[1] = 1
    exp_spf[1] = 0
    powerful[1] = 1
    
    for i in range(2, LIMIT + 1):
        if spf[i] == 0:
            spf[i] = i
            exp_spf[i] = 1
            powerful[i] = 1
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > LIMIT:
                break
            
            spf[i * p] = p
            
            if p == spf[i]:
                exp_spf[i * p] = exp_spf[i] + 1
                if exp_spf[i] == 1:
                    # p^1 -> p^2, so multiply by p^2
                    powerful[i * p] = powerful[i] * p * p
                else:
                    # p^k -> p^(k+1), so multiply by p
                    powerful[i * p] = powerful[i] * p
                break
            else:
                # p < spf[i], new prime factor with exp 1
                exp_spf[i * p] = 1
                powerful[i * p] = powerful[i]
    
    # Now find the n in [N_MIN, N_MAX] that maximizes powerful[n] * powerful[n+1]
    best_n = N_MIN
    best_val = powerful[N_MIN] * powerful[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = powerful[n] * powerful[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline - 2.0:
            break
    
    return best_n
# EVOLVE-BLOCK-END
