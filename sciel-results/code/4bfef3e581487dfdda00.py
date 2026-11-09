"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    MAX_VAL = N + 1  # We need powerful_part for n and n+1 where n <= 1000000
    
    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    # Using a list-based approach which is faster than numpy for this pattern
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number 1..MAX_VAL
    # powerful_part[i] = product of p^e for all prime factors p of i with exponent e >= 2
    # Using direct factorization with SPF, which is O(log n) per number.
    # This is robust and avoids the complexity/bugs of the sieve-based accumulation.
    
    powerful_part = [1] * (MAX_VAL + 1)
    
    # Time check every 10000 iterations to avoid overhead
    for i in range(2, MAX_VAL + 1):
        x = i
        p_prod = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count >= 2:
                p_prod *= (p ** count)
        powerful_part[i] = p_prod
        
        if i % 100000 == 0:
            if time.time() > deadline - 10:
                # If we're running out of time, we can break early, 
                # but we need the full range for the final max calculation.
                # However, 10^6 iterations is typically < 2 seconds in Python.
                # So this is just a safety guard.
                pass

    # 3. Find n in [10000, 1000000] maximizing powerful_part[n] * powerful_part[n+1]
    # We can use numpy for this step to speed up the max search, 
    # though a Python loop is also fast enough for 10^6 iterations.
    # Let's use numpy for elegance and speed in the reduction step.
    
    pp_np = np.array(powerful_part, dtype=np.int64)
    
    # products[i] = powerful_part[i] * powerful_part[i+1] for i in [LOW, N]
    # Note: powerful_part is 0-indexed in the list, but we want n in [10000, 1000000]
    # So we look at indices LOW to N in the list.
    # products will have length N - LOW + 1
    # product for n is pp[n] * pp[n+1]
    
    # Ensure we don't go out of bounds. pp has length MAX_VAL + 1 = 1000002
    # n goes up to 1000000, so n+1 is 1000001, which is within bounds.
    
    products = pp_np[LOW:N+1] * pp_np[LOW+1:N+2]
    
    idx = np.argmax(products)
    best_n = LOW + idx
    best_product = products[idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
