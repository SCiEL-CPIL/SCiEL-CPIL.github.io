import time
import math

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _divisors_from_factors_limited(factors, limit):
    """Generate all divisors of the number defined by factors that are < limit."""
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for p, e in zip(primes, exponents):
        current = []
        power = 1
        for i in range(e + 1):
            for d in divisors:
                val = d * power
                if val < limit:
                    current.append(val)
            power *= p
            if power >= limit:
                break
        divisors = current
    return divisors

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    MAX_DENOM = 10**12
    
    # Cache for factorizations
    factor_cache = {}
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # Pre-factorize n once
        if n not in factor_cache:
            factor_cache[n] = _factorize(n)
        fact_n = factor_cache[n]
        
        # x must satisfy: n/4 < x <= 3n/4 (from 3/x >= 4/n)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay - b)(az - b) = b^2
            # Let d = ay - b, then d | b^2, and d < b (since y < z)
            # y = (b + d) / a, z = (b + b^2/d) / a
            # Need: (b+d) % a == 0, (b + b^2//d) % a == 0
            # Need: x < y < z, y,z <= MAX_DENOM
            
            # Factor b = n * x to get divisors of b^2
            if x not in factor_cache:
                factor_cache[x] = _factorize(x)
            fact_x = factor_cache[x]
            
            # b^2 = n^2 * x^2
            fact_b2 = {}
            for p, e in fact_n.items():
                fact_b2[p] = fact_b2.get(p, 0) + 2 * e
            for p, e in fact_x.items():
                fact_b2[p] = fact_b2.get(p, 0) + 2 * e
            
            # Enumerate divisors d of b^2 with d < b
            divs = _divisors_from_factors_limited(fact_b2, b)
            
            for d in divs:
                # Check (b + d) % a == 0
                if (b + d) % a != 0:
                    continue
                y = (b + d) // a
                if y <= x:
                    continue
                # Check z
                b2 = b * b
                if b2 % d != 0:
                    continue
                z_val = b2 // d
                if (b + z_val) % a != 0:
                    continue
                z = (b + z_val) // a
                if z <= y:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        
        results[idx] = found
    
    return results
