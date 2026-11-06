import time
import math
from math import gcd
from collections import defaultdict

def factorize(num):
    """Return prime factorization of num as dict {prime: exponent}"""
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

def generate_divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict"""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    answers = [None] * 254
    BOUND = 10**12
    
    def remaining_time():
        return deadline - time.time()
    
    for n in range(3, 257):
        idx = n - 3
        if remaining_time() < 0.5:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        solved = False
        n_start = time.time()
        time_budget_n = min(3.0, remaining_time() * 0.12)
        
        for x in range(x_min, min(x_max + 1, x_min + 300)):
            if time.time() - n_start > time_budget_n or remaining_time() < 0.5:
                break
            
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # We need d = a*k - b to divide b^2, d > 0, d ≡ -b (mod a)
            # k = (d + b) / a must be integer, so d ≡ -b (mod a)
            # Also k > x, so d + b > a*x => d > a*x - b = 4x^2 - nx - nx = 4x^2 - 2nx
            # And k < 2b/a, so d + b < 2b => d < b
            
            # So d must be a divisor of b^2 such that:
            # 1. d > 0
            # 2. d < b  (from k < 2b/a => d < b)
            # 3. d ≡ -b (mod a)
            # 4. k = (d+b)/a > x, i.e., d > a*x - b
            # 5. z = (b^2/d + b)/a is integer, > k, and all distinct
            
            # Factor b to get divisors of b^2
            b_factors = factorize(b)
            # b^2 has exponents doubled
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            divisors_b2 = generate_divisors_from_factors(b2_factors)
            
            # Filter divisors: d > 0, d < b, d ≡ -b (mod a)
            neg_b_mod_a = (-b) % a
            candidates = []
            for d in divisors_b2:
                if d <= 0 or d >= b:
                    continue
                if d % a != neg_b_mod_a:
                    continue
                # k = (d + b) / a
                if (d + b) % a != 0:
                    continue
                k = (d + b) // a
                if k <= x:
                    continue
                # z = (b^2/d + b) / a
                e = b * b // d
                if (e + b) % a != 0:
                    continue
                z = (e + b) // a
                y = k
                if z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if x > BOUND or y > BOUND or z > BOUND:
                    continue
                candidates.append((x, y, z))
            
            if candidates:
                # Verify the first candidate
                x_c, y_c, z_c = candidates[0]
                lhs = n * (y_c * z_c + x_c * z_c + x_c * y_c)
                rhs = 4 * x_c * y_c * z_c
                if lhs == rhs:
                    found = [x_c, y_c, z_c]
                    solved = True
                    break
        
        answers[idx] = found
    
    return answers
