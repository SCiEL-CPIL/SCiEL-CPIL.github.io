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
    
    for i in range(2, LIMIT + 1):
        if spf[i] == 0:
            # i is prime
            spf[i] = i
            exp_spf[i] = 1
            powerful[i] = 1  # prime itself has no prime power with e>=2
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > LIMIT:
                break
            spf[i * p] = p
            if p == spf[i]:
                exp_spf[i * p] = exp_spf[i] + 1
                # powerful[i*p]: if exp_spf[i*p] >= 2, include p^(exp_spf[i*p])
                # powerful[i] already has the powerful part of i
                # We need to be careful: powerful[i] = product of q^e for q^e || i with e>=2
                # For i*p where p = spf[i], the exponent of p increases by 1
                # If exp_spf[i] == 1, then p^1 was not in powerful[i], but p^2 is in powerful[i*p]
                # If exp_spf[i] >= 2, then p^exp_spf[i] was in powerful[i], now p^(exp_spf[i]+1) is
                if exp_spf[i * p] >= 2:
                    if exp_spf[i] >= 2:
                        # p^exp_spf[i] was in powerful[i], replace with p^(exp_spf[i]+1)
                        powerful[i * p] = powerful[i] // (p ** exp_spf[i]) * (p ** (exp_spf[i] + 1))
                    else:
                        # exp_spf[i] == 1, so p was not in powerful[i], add p^2
                        powerful[i * p] = powerful[i] * (p * p)
                else:
                    # exp_spf[i*p] == 1, so p not included
                    powerful[i * p] = powerful[i]
                break
            else:
                # p < spf[i], so p is a new prime factor with exponent 1
                exp_spf[i * p] = 1
                powerful[i * p] = powerful[i]
        
        if time.time() > deadline - 10.0:
            # Fallback: use trial division for remaining numbers
            for j in range(i + 1, LIMIT + 1):
                if spf[j] == 0:
                    # Compute powerful part by trial division
                    temp = j
                    pp_val = 1
                    d = 2
                    while d * d <= temp:
                        if temp % d == 0:
                            exp = 0
                            while temp % d == 0:
                                temp //= d
                                exp += 1
                            if exp >= 2:
                                pp_val *= d ** exp
                        d += 1
                    if temp > 1:
                        # temp is prime with exponent 1
                        pass
                    powerful[j] = pp_val
                    spf[j] = temp if temp > 1 else 1
            break
    
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
