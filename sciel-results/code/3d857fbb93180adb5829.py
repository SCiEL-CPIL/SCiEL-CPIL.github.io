import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need to compute up to n+1 where n can be MAX_N
    
    # 1. Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    # Only primes p where p^2 <= LIMIT can contribute to powerful_part
    limit_sqrt = int(math.isqrt(LIMIT))
    is_prime = [True] * (limit_sqrt + 1)
    is_prime[0] = False
    if limit_sqrt >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_sqrt)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit_sqrt + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Initialize powerful_part array using NumPy for speed
    # dtype=np.int64 is sufficient for values up to ~10^12 product, 
    # but individual P(n) <= n <= 10^6, so int32 would suffice for P(n), 
    # but product needs int64. We keep P(n) in int64 to avoid casting during multiplication.
    powerful_part = np.ones(LIMIT + 1, dtype=np.int64)
    
    # 3. For each prime, process its powers
    for p in primes:
        p_sq = p * p
        if p_sq > LIMIT:
            break
        
        p_pow = p_sq
        while p_pow <= LIMIT:
            next_pow = p_pow * p
            
            # We want to multiply powerful_part[m] by p_pow
            # for all m such that m is a multiple of p_pow but NOT a multiple of next_pow.
            
            # Using NumPy slicing/boolean indexing
            # Indices divisible by p_pow
            # We can create a mask or just use advanced indexing.
            # mask = np.zeros(LIMIT + 1, dtype=bool)
            # mask[p_pow::p_pow] = True
            # if next_pow <= LIMIT:
            #     mask[next_pow::next_pow] = False
            # powerful_part[mask] *= p_pow
            
            # More efficient:
            # Get indices
            # But creating a full mask of size 10^6 for each prime power might be slow if done many times.
            # Number of prime powers is small.
            # Let's try the mask approach first as it's clean.
            
            mask = np.zeros(LIMIT + 1, dtype=bool)
            mask[p_pow::p_pow] = True
            if next_pow <= LIMIT:
                mask[next_pow::next_pow] = False
            
            powerful_part[mask] *= p_pow
            
            if next_pow > LIMIT:
                break
            p_pow = next_pow
            
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return MIN_N
    
    # 4. Scan all n in [MIN_N, MAX_N] and find max powerful_part[n] * powerful_part[n+1]
    # powerful_part[n] for n in [MIN_N, MAX_N]
    # powerful_part[n+1] for n+1 in [MIN_N+1, MAX_N+1]
    
    # Slice the arrays
    # n goes from MIN_N to MAX_N
    # p_n = powerful_part[MIN_N : MAX_N + 1]
    # p_np1 = powerful_part[MIN_N + 1 : MAX_N + 2]
    
    p_n = powerful_part[MIN_N : MAX_N + 1]
    p_np1 = powerful_part[MIN_N + 1 : MAX_N + 2]
    
    products = p_n * p_np1
    
    # Find index of maximum product
    max_idx = np.argmax(products)
    best_n = MIN_N + max_idx
    
    # Safety check
    if best_n < MIN_N or best_n > MAX_N:
        best_n = MIN_N
        
    return int(best_n)
