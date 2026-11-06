import time
from functools import lru_cache

def factorize(n):
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

def generate_divisors_from_factors(factors):
    divs = [1]
    for p, e in factors.items():
        current_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                current_divs.append(val)
                val *= p
        divs = current_divs
    return divs

@lru_cache(maxsize=None)
def get_divisors_of_sq(n):
    """Get divisors of n^2, cached."""
    fac = factorize(n)
    fac_sq = {p: 2*e for p, e in fac.items()}
    return tuple(generate_divisors_from_factors(fac_sq))

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        
        # Try x values in a bounded range
        for x in range(x_min, 5 * n + 10):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            B_sq = B * B
            
            # Get divisors of B^2 (cached)
            divisors = get_divisors_of_sq(B)
            
            for u in divisors:
                if u > B:
                    continue
                if (u + B) % A != 0:
                    continue
                
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if x == y or x == z or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                if y < 1 or z < 1:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
