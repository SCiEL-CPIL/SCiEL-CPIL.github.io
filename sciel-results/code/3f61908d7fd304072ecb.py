import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need powerful_part for n and n+1, so up to 1000001
    
    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Collect primes
    primes = []
    for i in range(2, MAX_VAL + 1):
        if spf[i] == i:
            primes.append(i)
    
    # 3. Compute powerful_part using sieve over prime powers
    # For each prime p and exponent e >= 2:
    #   For each multiple m of p^e:
    #     If m is NOT divisible by p^(e+1), then powerful_part[m] *= p^e
    powerful_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        power = p * p  # p^2
        while power <= MAX_VAL:
            next_power = power * p
            if next_power > MAX_VAL:
                # All multiples of power have exponent exactly e (since p^(e+1) > MAX_VAL)
                for m in range(power, MAX_VAL + 1, power):
                    powerful_part[m] *= power
            else:
                # Only multiples of power that are NOT multiples of next_power
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        powerful_part[m] *= power
            
            power = next_power
            
            if time.time() > deadline - 5.0:
                break
    
    # 4. Find n in [10000, 1000000] maximizing powerful_part[n] * powerful_part[n+1]
    max_val = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
