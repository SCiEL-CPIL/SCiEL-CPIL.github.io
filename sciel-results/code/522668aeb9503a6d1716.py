"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    BOUND = 1000000000000
    
    def factorize(num):
        """Factorize a number into prime factors."""
        factors = {}
        f = 2
        while f * f <= num:
            while num % f == 0:
                factors[f] = factors.get(f, 0) + 1
                num //= f
            f += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    def get_divisors_from_factorization(factors):
        """Generate all divisors from prime factorization."""
        primes = list(factors.keys())
        exps = [factors[p] for p in primes]
        
        divisors = [1]
        for p, e in zip(primes, exps):
            new_divs = []
            for d in divisors:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divisors = new_divs
        return divisors
    
    def find_via_divisors(n, x, a, b):
        """Find y,z via divisor enumeration of b^2."""
        # (ay - b)(az - b) = b^2
        # Let d = ay - b, then az - b = b^2/d
        # y = (d + b) / a, z = (b^2/d + b) / a
        b2 = b * b
        
        # Factor b to get divisors of b^2
        factors_b = factorize(b)
        # b^2 has doubled exponents
        factors_b2 = {p: 2*e for p, e in factors_b.items()}
        
        divisors = get_divisors_from_factorization(factors_b2)
        
        # Sort divisors to find solutions with reasonable y first
        # We want y > x, z > y (or at least distinct)
        # y = (d + b) / a, so d ≈ a*y - b
        # For y > x: d > a*x - b = a*x - n*x = x*(a-n) = x*(4x-2n)
        
        for d in divisors:
            if d + b <= 0:
                continue
            if (d + b) % a != 0:
                continue
            y = (d + b) // a
            if y <= 0 or y > BOUND:
                continue
            
            if b2 % d != 0:
                continue
            bz = b2 // d
            if bz + b <= 0:
                continue
            if (bz + b) % a != 0:
                continue
            z = (bz + b) // a
            if z <= 0 or z > BOUND:
                continue
            
            # Check distinctness
            if x == y or y == z or x == z:
                continue
            
            # Ensure positive and valid
            if y > 0 and z > 0:
                return [x, y, z]
        
        return None
    
    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # Strategy 1: Try common parametric constructions
        # For 4/n, try x = n//2 + 1, x = n//3 + 1, etc.
        # Actually, let's try a few small offsets from n/4
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Quick parametric: try x values that make 4x-n a small number
        # 4x - n = 1, 2, 3, ... small values often lead to nice solutions
        for offset in range(0, 10):
            x = x_min + offset
            if x > x_max:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Direct scan for y (fast for small ranges)
            y_min = max(x + 1, b // a + 1)
            y_max = (2 * b) // a
            
            if y_max - y_min < 1000:  # Range is small, direct scan is fast
                for y in range(y_min, y_max + 1):
                    den = a * y - b
                    if den <= 0:
                        continue
                    num = b * y
                    if num % den == 0:
                        z = num // den
                        if z > y and z <= BOUND and z != x:
                            found = [x, y, z]
                            break
                if found:
                    break
            else:
                # Range is large, use divisor enumeration
                result = find_via_divisors(n, x, a, b)
                if result:
                    found = result
                    break
        
        # If not found, expand x search
        if not found:
            for x in range(x_min, min(x_min + 500, x_max + 1)):
                if time.time() > deadline:
                    break
                    
                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                
                y_min = max(x + 1, b // a + 1)
                y_max = (2 * b) // a
                
                if y_max < y_min:
                    continue
                
                # For small ranges, direct scan; for large, divisor enumeration
                if y_max - y_min < 500:
                    for y in range(y_min, y_max + 1):
                        den = a * y - b
                        if den <= 0:
                            continue
                        num = b * y
                        if num % den == 0:
                            z = num // den
                            if z > y and z <= BOUND and z != x:
                                found = [x, y, z]
                                break
                else:
                    result = find_via_divisors(n, x, a, b)
                    if result:
                        found = result
                        break
                
                if found:
                    break
        
        answers[idx] = found
        
    return answers
# EVOLVE-BLOCK-END
