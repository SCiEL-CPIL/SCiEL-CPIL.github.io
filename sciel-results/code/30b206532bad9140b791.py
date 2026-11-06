"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

# Sieve primes up to 256 (covers all prime factors of b = n*x for n,x <= 256)
def _make_sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

_PRIMES = _make_sieve(256)

def _factorize(n):
    """Return list of (prime, exponent) for n. n must have all prime factors <= 256."""
    factors = []
    for p in _PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors.append((p, e))
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a list of (prime, exponent) pairs."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution_for_n(n, deadline):
    """Find distinct positive integers x,y,z with 1/x+1/y+1/z=4/n. Return [x,y,z] or None."""
    # x must satisfy n/4 < x < 3n/4 (from 1/x < 4/n and 4/n < 3/x)
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    # Try small x first: small b = n*x means small b^2, fast factorization,
    # and small x gives large a/b ratio making y range manageable.
    # Also try a few strategic x values.
    x_candidates = list(range(x_min, x_max + 1))
    
    for x in x_candidates:
        if time.time() > deadline:
            return None
        a = 4 * x - n
        b = n * x
        if a <= 0:
            continue
        
        # We need (a*y - b)(a*z - b) = b^2
        # d = a*y - b, so d | b^2, d < b, (d + b) % a == 0, y = (d+b)//a > x
        # z = (b^2//d + b) // a, need z > y, z != x
        
        # Factor b^2
        b_sq = b * b
        factors = _factorize(b_sq)
        divs = _divisors_from_factors(factors)
        
        # Filter: d < b, (d + b) % a == 0
        neg_b_mod_a = (-b) % a
        found = None
        for d in divs:
            if d >= b:
                continue
            if (d + b) % a != 0:
                continue
            y = (d + b) // a
            if y <= x:
                continue
            # Check z
            rem = b_sq // d
            if (rem + b) % a != 0:
                continue
            z = (rem + b) // a
            if z <= y:
                continue
            if z == x or y == x:
                continue
            # Verify equation by cross multiplication
            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                if x <= 1000000000000 and y <= 1000000000000 and z <= 1000000000000:
                    found = [x, y, z]
                    break
        if found:
            return found
    
    return None

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, deadline)
    
    return results
# EVOLVE-BLOCK-END
