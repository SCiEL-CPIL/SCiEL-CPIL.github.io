import sys
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = 80.0  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need powerful_part for n and n+1 where n <= 1000000
    
    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    # spf[x] = smallest prime factor of x
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number 1..MAX_VAL
    # powerful_part[i] = product of p^e for all prime factors p of i with exponent e >= 2
    # Using the sieve approach from inspiration 1: for each prime p and exponent e >= 2,
    # multiply p^e into powerful_part[m] for m divisible by p^e but NOT by p^(e+1)
    
    # First, collect all primes
    primes = []
    for i in range(2, MAX_VAL + 1):
        if spf[i] == i:
            primes.append(i)
    
    powerful_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        power = p * p  # p^2
        while power <= MAX_VAL:
            next_power = power * p
            # For multiples of 'power' that are NOT multiples of 'next_power',
            # the exponent of p is exactly e
            if next_power > MAX_VAL:
                # All multiples of power have exactly this exponent
                for m in range(power, MAX_VAL + 1, power):
                    powerful_part[m] *= power
            else:
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        powerful_part[m] *= power
            power = next_power
            
            # Time check
            if time.time() - start_time > limit_time - 5:
                break
    
    # 3. Find n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    max_val = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
        
        # Time check
        if time.time() - start_time > limit_time:
            break
            
    return best_n
# EVOLVE-BLOCK-END
