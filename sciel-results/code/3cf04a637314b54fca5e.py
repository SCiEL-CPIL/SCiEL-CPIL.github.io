import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # Leave margin for overhead
    
    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation using precomputed primes
    def phi(x):
        if x == 1:
            return 1
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    # 3. Search for a run of length 200 with distinct totients
    # We want to maximize k. Max k is 200.
    # We check if a run of length 200 exists.
    # Given the analysis, it's highly likely for large s.
    
    best_k = 1
    best_s = 1
    
    # First, try some deterministic starting points that are likely to work
    # Large even numbers tend to have phi values that are well-distributed
    test_starts = [
        10**12 - 200,  # Near the upper bound
        10**11,
        10**10,
        10**9,
        5 * 10**11,
        3 * 10**11,
        7 * 10**10,
    ]
    
    for s in test_starts:
        if s < 1 or s + 199 > 10**12:
            continue
        seen = set()
        k = 0
        valid = True
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                valid = False
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
    
    # If deterministic didn't find 200, try random sampling
    random.seed(42)
    attempts = 0
    while time.time() - start_time < LIMIT:
        s = random.randint(10**9, 10**12 - 199)
        
        seen = set()
        k = 0
        valid = True
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                valid = False
                break
            seen.add(val)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
        
        attempts += 1
        if attempts > 200:
            break

    # Fallback: if we didn't find 200, try a structured search near 10^12?
    # Or just return what we have.
    
    return {'start': best_s, 'length': best_k}
