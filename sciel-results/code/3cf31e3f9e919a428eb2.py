import time
from math import gcd

def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime divisor pairs."""
    if n < 2:
        return 0
    
    # Factorize n
    factors = {}
    temp_n = n
    p = 2
    while p * p <= temp_n:
        if temp_n % p == 0:
            count = 0
            while temp_n % p == 0:
                count += 1
                temp_n //= p
            factors[p] = count
        p += 1 if p == 2 else 2
    if temp_n > 1:
        factors[temp_n] = 1
        
    # Generate divisors
    divs = [1]
    for p, exp in factors.items():
        new_divs = []
        p_power = 1
        for _ in range(exp + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
        
    divs.sort()
    
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _get_primorial(k):
    """Return product of first k primes."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    prod = 1
    for i in range(k):
        prod *= primes[i]
    return prod

def _generate_smart_candidates():
    """Generate a small set of high-potential candidates."""
    candidates = set()
    
    # 1. Primorials
    for k in range(1, 16):
        p_k = _get_primorial(k)
        if p_k <= 10**18:
            candidates.add(p_k)
            
    # 2. P15 multiplied by small integers
    p15 = _get_primorial(15)
    for i in range(2, 100):
        val = p15 * i
        if val <= 10**18:
            candidates.add(val)
        else:
            break
            
    # 3. P14 multiplied by 2^a 3^b 5^c
    p14 = _get_primorial(14)
    limit = 10**18 // p14
    
    for a in range(0, 11):
        v2 = 2 ** a
        if v2 > limit: break
        for b in range(0, 11):
            v3 = 3 ** b
            if v2 * v3 > limit: break
            for c in range(0, 11):
                v5 = 5 ** c
                if v2 * v3 * v5 > limit: break
                val = p14 * v2 * v3 * v5
                if val <= 10**18:
                    candidates.add(val)
                    
    return candidates

def construct():
    """Entry point. Returns an integer n with 2 <= n <= 10^18 maximizing tau_perp."""
    deadline = time.time() + 110  # Safety margin
    
    candidates = _generate_smart_candidates()
    
    best_n = 2
    best_score = 0
    
    valid_candidates = [n for n in candidates if 2 <= n <= 10**18]
    
    for n in valid_candidates:
        if time.time() > deadline:
            break
        score = _tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
