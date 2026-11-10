# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need pp[MAX_N+1]
    
    # Build smallest prime factor sieve up to LIMIT
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [1, LIMIT]
    # pp[n] = product of p^e for each prime factor p with exponent e >= 2
    pp = [1] * (LIMIT + 1)
    
    # Iterate through primes
    # A number is prime if spf[x] == x
    for p in range(2, LIMIT + 1):
        if spf[p] != p:
            continue
        
        # Check if p^2 > LIMIT
        if p * p > LIMIT:
            break
            
        power = p * p
        next_power = power * p
        
        while power <= LIMIT:
            # For each multiple of 'power'
            # If it is NOT divisible by 'next_power', then exponent is exactly current k
            # (Wait, this logic assumes we are iterating k. 
            #  If m is divisible by p^k but not p^(k+1), we add p^k.
            #  This is correct.)
            
            for m in range(power, LIMIT + 1, power):
                if m % next_power != 0:
                    pp[m] *= power
            
            power = next_power
            next_power *= p
            
            # Time check
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break
    
    # Find n in [MIN_N, MAX_N] that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N + 1):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        
        # Time check
        if time.time() > deadline - 5:
            break
    
    return best_n
# EVOLVE-BLOCK-END
