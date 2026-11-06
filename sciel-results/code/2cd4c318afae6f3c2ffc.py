"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct denominators <= 10^12."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # safety margin before 90s kill
    MAX_DEN = 10**12
    
    def factorize(n):
        """Return dict of prime factors and exponents."""
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
    
    def divisors_from_factors(factors):
        """Generate all divisors from a factorization dict."""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                pk = 1
                for k in range(e + 1):
                    new_divs.append(d * pk)
                    pk *= p
            divs = new_divs
        return divs
    
    def get_divisors_b_sq(b):
        """Get divisors of b^2 by factoring b first."""
        factors = factorize(b)
        factors_sq = {p: 2*e for p, e in factors.items()}
        return divisors_from_factors(factors_sq)

    def find_solution(n):
        """Find distinct x<y<z with 1/x+1/y+1/z=4/n, denominators <= MAX_DEN."""
        x_min = n // 4 + 1
        x_upper = (3 * n) // 4 + 1  # tighter bound: x < 3n/4
        
        for x in range(x_min, x_upper + 1):
            if time.time() > deadline:
                return None
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need d = a*y - b to be a divisor of b^2.
            # Conditions on d:
            # 1. d > a*x - b (from y > x)
            # 2. d < b (from z > y, which implies y < 2b/a => d < b)
            # 3. (d + b) % a == 0 (so y is integer)
            # 4. z = b*y/d is integer and <= MAX_DEN
            
            d_min = a * x - b
            d_max = b - 1
            
            if d_min >= d_max:
                continue
            
            divs = get_divisors_b_sq(b)
            
            for d in divs:
                if d <= d_min or d >= d_max:
                    continue
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                # Check y > x is guaranteed by d > d_min
                # Check y < 2b/a is guaranteed by d < b
                # Check z
                if (b * y) % d != 0:
                    continue
                z = (b * y) // d
                if z <= y:
                    continue
                if z > MAX_DEN:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                return (x, y, z)
            
        return None
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        sol = find_solution(n)
        answers.append(sol)
    
    return answers
