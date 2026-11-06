import time
import math

def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divs = [1]
    for i, (p, e) in enumerate(zip(primes, exponents)):
        current = []
        pow_p = 1
        for k in range(e + 1):
            for d in divs:
                current.append(d * pow_p)
            pow_p *= p
        divs = current
    return divs

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        # Try direct formulas for even n
        if n % 2 == 0:
            if n % 4 == 0:
                x = n // 2
                y = 3 * n // 4
                z = 3 * n // 2
                # Check distinctness and bounds
                if x != y and x != z and y != z and x <= 10**12 and y <= 10**12 and z <= 10**12:
                    results[idx] = [x, y, z]
                    continue
            else: # n % 4 == 2
                x = n // 2
                y = n // 2 + 1
                z = x * y
                if x != y and x != z and y != z and x <= 10**12 and y <= 10**12 and z <= 10**12:
                    results[idx] = [x, y, z]
                    continue
        
        # Fallback to divisor search for odd n or if formula failed
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # Factor B, then get divisors of B^2
            bf = _factorize(B)
            # Double exponents for B^2
            bf2 = {p: 2*e for p, e in bf.items()}
            divs = _divisors_from_factors(bf2)
            
            for dd in divs:
                if (dd + B) % A != 0:
                    continue
                y = (dd + B) // A
                if y <= 0:
                    continue
                rem = B2 // dd
                if (rem + B) % A != 0:
                    continue
                z = (rem + B) // A
                if z <= 0:
                    continue
                if x == y or x == z or y == z:
                    continue
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        results[idx] = found
    
    return results
