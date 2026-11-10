import time
from math import gcd

# EVOLVE-BLOCK-START
def _factorize(n):
    """Return dict of prime -> exponent for n."""
    if n <= 1:
        return {}
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
    exponents = list(factors.values())
    divs = [1]
    for i, p in enumerate(primes):
        current = list(divs)
        mul = 1
        for _ in range(exponents[i]):
            mul *= p
            for d in current:
                divs.append(d * mul)
    return divs

def _merge_factors(f1, f2):
    """Merge two factor dicts."""
    merged = dict(f1)
    for p, e in f2.items():
        merged[p] = merged.get(p, 0) + e
    return merged

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s wall-clock
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        n2 = n * n
        n2_factors = _factorize(n2)
        
        # x range: n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            d = 4 * x - n
            if d <= 0:
                continue
            
            # We need divisors m of n^2 * x such that:
            # m > n*x, m ≡ -n*x (mod d), and y = (m + n*x)/d > x, z = n*x*y/m > y
            
            # Factorize n^2 * x
            x_factors = _factorize(x)
            target_factors = _merge_factors(n2_factors, x_factors)
            target = n2 * x
            nx = n * x
            
            # Generate divisors of target that are > nx
            # For efficiency, we can filter during generation or after
            divs = _divisors_from_factors(target_factors)
            
            # Filter: m > nx and m ≡ -nx (mod d)
            # -nx mod d = d - (nx % d) if nx % d != 0, else 0
            rem_needed = (-nx) % d
            
            for m in divs:
                if m <= nx:
                    continue
                if m % d != rem_needed:
                    continue
                
                y = (m + nx) // d
                if y <= x:
                    continue
                
                # z = n*x*y / m must be integer
                if (n * x * y) % m != 0:
                    continue
                z = (n * x * y) // m
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                
                # Verify via cross multiplication
                lhs = n * y * z + n * x * z + n * x * y
                rhs = 4 * x * y * z
                if lhs != rhs:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
