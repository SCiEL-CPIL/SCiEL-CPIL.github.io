import math
import random

def construct():
    # Approach: Randomized search for a starting point s.
    # This differs from the deterministic offset scan of the seed.
    
    # 1. Generate primes up to 10^6 for factorization
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    max_k = 200
    # Search in a high range to ensure distinctness is highly probable
    # Using random sampling instead of linear offsets
    # Range: 10^10 to 10^12
    lower_bound = 10**10
    upper_bound = 10**12
    
    # Try a few random starting points
    for _ in range(100):
        s = random.randint(lower_bound, upper_bound - max_k)
        
        # Check if s + max_k - 1 <= 10^12
        if s + max_k - 1 > 10**12:
            continue
            
        seen = set()
        distinct = True
        
        for i in range(max_k):
            n = s + i
            v = phi(n)
            if v in seen:
                distinct = False
                break
            seen.add(v)
            
        if distinct:
            return {'start': s, 'length': max_k}
            
    # Fallback: Deterministic search if random fails (unlikely)
    base_s = 10**11
    for offset in range(0, 100):
        s = base_s + offset * 1000
        if s + max_k - 1 > 10**12:
            break
        seen = set()
        distinct = True
        for i in range(max_k):
            n = s + i
            v = phi(n)
            if v in seen:
                distinct = False
                break
            seen.add(v)
        if distinct:
            return {'start': s, 'length': max_k}
            
    return {'start': 1, 'length': 1}
