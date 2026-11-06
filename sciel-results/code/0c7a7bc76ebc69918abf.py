import time
import math

def _factorize(num):
    """Return prime factorization of num as dict {prime: exponent}."""
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

def _divisors_from_factorization(factors):
    """Generate all divisors from a prime factorization dict."""
    divisors = [1]
    for prime, exp in factors.items():
        new_divisors = []
        power = 1
        for e in range(exp + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= prime
        divisors = new_divisors
    return divisors

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_start = n // 4 + 1
        x_end = (3 * n + 3) // 4
        
        # Try a limited set of x values: prioritize smaller x, add some spread
        max_x_tries = 30
        x_candidates = list(range(x_start, min(x_start + max_x_tries, x_end + 1)))
        if x_end - x_start > max_x_tries:
            step = max(1, (x_end - x_start) // 5)
            for i in range(1, 6):
                x_val = x_start + i * step
                if x_val <= x_end and x_val not in x_candidates:
                    x_candidates.append(x_val)
        
        for x in x_candidates:
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Factorize b to efficiently enumerate divisors of b^2
            factors_b = _factorize(b)
            factors_b_sq = {p: 2 * e for p, e in factors_b.items()}
            divs = _divisors_from_factorization(factors_b_sq)
            
            for u in divs:
                v = b_sq // u
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND or x > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
    
        answers[idx] = found
    
    return answers
