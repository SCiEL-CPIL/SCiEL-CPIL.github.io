import math
import random

def construct():
    # 1. Generate primes up to 10^6 using Sieve of Eratosthenes
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]

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

    # 2. Search for a run of length 200 using randomized sampling
    # This approach samples different starting points to ensure robustness
    # against any specific local patterns, though collisions are rare.
    max_k = 200
    upper_bound = 10**12
    lower_bound = 10**11
    
    # Try several random starting points
    for _ in range(50):
        s = random.randint(lower_bound, upper_bound - max_k)
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
            
    # Fallback: Deterministic check at a fixed high point
    s = 10**11
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

    # Ultimate fallback: return a valid small run
    return {'start': 1, 'length': 1}
