import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need powerful_part for n and n+1 where n <= 1000000
    
    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number 1..MAX_VAL using DP
    # powerful_part[i] = product of p^e for all prime factors p of i with exponent e >= 2
    powerful_part = [1] * (MAX_VAL + 1)
    # p_power[i] will store the highest power of spf[i] that divides i
    p_power = [1] * (MAX_VAL + 1)
    
    # Base case: 1
    powerful_part[1] = 1
    p_power[1] = 1
    
    for i in range(2, MAX_VAL + 1):
        p = spf[i]
        m = i // p
        
        # Determine p_power[i]
        if spf[m] == p:
            # p divides m, so the exponent of p in i is exponent in m + 1
            p_power[i] = p_power[m] * p
        else:
            # p does not divide m, so the exponent of p in i is 1
            p_power[i] = p
        
        rem = i // p_power[i]
        
        if p_power[i] > p:
            # Exponent >= 2, include p^e in powerful part
            powerful_part[i] = p_power[i] * powerful_part[rem]
        else:
            # Exponent == 1, do not include p^e
            powerful_part[i] = powerful_part[rem]
            
        # Time check (unlikely to be needed for N=1e6, but good practice)
        if time.time() > deadline:
            # Fallback if we are extremely slow (should not happen)
            break

    # 3. Find n in [10000, 1000000] maximizing powerful_part[n] * powerful_part[n+1]
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
