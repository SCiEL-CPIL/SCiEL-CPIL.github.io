import math
import time

def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 80  # 80s safety margin out of 90s
    
    def factorize(num):
        """Return dict of prime -> exponent for num."""
        if num <= 1:
            return {}
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 1)
        return factors
    
    def divisors_from_factors(factors):
        """Generate all divisors from prime factorization dict."""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            pe = 1
            for i in range(e + 1):
                for d in divs:
                    new_divs.append(d * pe)
                pe *= p
            divs = new_divs
        return divs
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        # Try x values starting from x_min
        for x in range(x_min, x_min + 200):  # try up to 200 values of x
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Factorize b_sq
            fac = factorize(b_sq)
            divs = divisors_from_factors(fac)
            
            for d in divs:
                # y = (d + b) / a, z = (b_sq/d + b) / a
                num_y = d + b
                num_z = b_sq // d + b  # d divides b_sq so this is exact
                if num_y % a != 0 or num_z % a != 0:
                    continue
                y = num_y // a
                z = num_z // a
                if y <= 0 or z <= 0:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                # Ensure distinctness: x, y, z all different
                if x == y or x == z or y == z:
                    continue
                # Verify the equation
                # 1/x + 1/y + 1/z == 4/n
                # Check: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
