import time
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # 55s limit to be safe within 60s/90s constraints
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # Need to check n+1 up to 1000001
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    # Optimization: only iterate up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            # Mark multiples
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute powerful part for each number
    # pow_part[x] is the product of p^e for all p^e || x with e >= 2
    pow_part = [1] * (MAX_VAL + 1)
    
    for x in range(2, MAX_VAL + 1):
        n = x
        pp = 1
        while n > 1:
            p = spf[n]
            count = 0
            power_val = 1
            while n % p == 0:
                n //= p
                count += 1
                power_val *= p
            if count >= 2:
                pp *= power_val
        pow_part[x] = pp
        
    # Check time
    if time.time() - start_time > deadline:
         # Fallback if calculation is unexpectedly slow
         # Return a known good candidate or seed
         return 10000

    # 3. Find n that maximizes pow_part[n] * pow_part[n+1]
    max_product = 0
    best_n = 10000
    
    for n in range(10000, MAX_N + 1):
        prod = pow_part[n] * pow_part[n + 1]
        if prod > max_product:
            max_product = prod
            best_n = n
            
    return best_n
