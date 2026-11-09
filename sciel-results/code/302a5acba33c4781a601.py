import math
import time

def _factorize(n):
    """Return prime factorization of n as a dictionary {prime: exponent}."""
    factors = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    d = 3
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a factorization dictionary."""
    divisors = [1]
    for prime, exp in factors.items():
        current_divs = []
        power = 1
        for _ in range(exp + 1):
            for d in divisors:
                current_divs.append(d * power)
            power *= prime
        divisors = current_divs
    return divisors

def construct():
    deadline = time.time() + 80
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        if x_max < x_min:
            x_max = x_min
            
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            b2 = b * b
            
            # Factorize b
            factors_b = _factorize(b)
            # Factorize b^2: exponents are doubled
            factors_b2 = {p: e * 2 for p, e in factors_b.items()}
            
            # Generate divisors of b^2
            divs = _divisors_from_factors(factors_b2)
            
            for d in divs:
                d1 = d
                d2 = b2 // d
                
                # Check congruence: d1 ≡ -b (mod a) and d2 ≡ -b (mod a)
                # Note: a > 0 is guaranteed by x_min
                if (d1 + b) % a == 0 and (d2 + b) % a == 0:
                    y = (d1 + b) // a
                    z = (d2 + b) // a
                    
                    # Check distinctness and positivity
                    if y > 0 and z > 0 and x != y and y != z and x != z:
                        # Check bound
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
            if found is not None:
                break
                
        answers[idx] = found
        
    return answers
