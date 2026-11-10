import math

def construct():
    # We want to find s, k such that phi(s)...phi(s+k-1) are distinct.
    # Max k is 200.
    # We search in a high range to minimize collision probability.
    
    # 1. Generate primes up to 10^6
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

    # 2. Search for a run of length 200
    # Start from a large number to ensure distinctness is highly probable
    # 10^11 is well below 10^12
    base_s = 10**11
    max_k = 200
    
    # Try a few starting points
    for offset in range(0, 1000):
        s = base_s + offset * 1000
        if s + max_k - 1 > 10**12:
            break
            
        vals = []
        distinct = True
        seen = set()
        
        for i in range(max_k):
            n = s + i
            v = phi(n)
            if v in seen:
                distinct = False
                break
            seen.add(v)
            vals.append(v)
            
        if distinct:
            return {'start': s, 'length': max_k}
            
    # Fallback: if no 200 found (extremely unlikely), try to find a shorter one
    # Just return a valid small run if needed, but 200 should be found.
    # For safety, return the first one we could verify, or a default.
    # Let's just return a default valid one if the loop fails.
    return {'start': 1, 'length': 1}
