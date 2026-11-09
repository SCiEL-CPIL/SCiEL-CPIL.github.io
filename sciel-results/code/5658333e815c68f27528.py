import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need to compute up to n+1 where n can be MAX_N
    
    # 1. Sieve of Eratosthenes to find primes up to LIMIT
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i * i, LIMIT + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Initialize powerful_part array
    powerful_part = [1] * (LIMIT + 1)
    
    # 3. For each prime, process its powers
    for p in primes:
        p_sq = p * p
        if p_sq > LIMIT:
            break
        
        p_pow = p_sq
        while p_pow <= LIMIT:
            next_pow = p_pow * p
            # For all multiples m of p_pow
            # If m is NOT a multiple of next_pow, then the exponent of p in m is exactly e (where p_pow = p^e)
            # So we multiply powerful_part[m] by p_pow
            
            # Optimization: We can iterate multiples.
            # If next_pow > LIMIT, then NO multiple of p_pow is a multiple of next_pow (within range)
            # So we can just multiply all multiples.
            if next_pow > LIMIT:
                for m in range(p_pow, LIMIT + 1, p_pow):
                    powerful_part[m] *= p_pow
            else:
                # We need to check if m % next_pow != 0
                # m = k * p_pow
                # m % next_pow == 0  <=>  k * p_pow % (p * p_pow) == 0  <=>  k % p == 0
                # So we skip multiples where k is a multiple of p.
                # i.e., skip m = k * p_pow where k is a multiple of p.
                # These are multiples of p * p_pow = next_pow.
                
                # We can iterate k from 1 to LIMIT // p_pow
                # and skip if k % p == 0
                # This is equivalent to iterating m and checking m % next_pow != 0
                # But checking k % p == 0 might be slightly faster?
                # Let's stick to m % next_pow != 0 for clarity, or use the k loop.
                
                # Using k loop:
                # max_k = LIMIT // p_pow
                # for k in range(1, max_k + 1):
                #     if k % p != 0:
                #         powerful_part[k * p_pow] *= p_pow
                
                # This avoids modulo on large numbers m, using modulo on smaller k.
                max_k = LIMIT // p_pow
                for k in range(1, max_k + 1):
                    if k % p != 0:
                        powerful_part[k * p_pow] *= p_pow
            
            p_pow = next_pow
            
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return MIN_N
    
    # 4. Scan all n in [MIN_N, MAX_N] and find max powerful_part[n] * powerful_part[n+1]
    best_product = 0
    best_n = MIN_N
    
    for n in range(MIN_N, MAX_N + 1):
        product = powerful_part[n] * powerful_part[n + 1]
        if product > best_product:
            best_product = product
            best_n = n
    
    # Safety check
    if best_n < MIN_N or best_n > MAX_N:
        best_n = MIN_N
    
    return best_n
