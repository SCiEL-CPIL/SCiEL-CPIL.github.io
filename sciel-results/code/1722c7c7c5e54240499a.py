"""Improved program for distinct three-term unit-fraction representations of 4/n."""
import time
import math

# EVOLVE-BLOCK-START

def _factorize(n):
    """Return prime factorization as dict {prime: exponent}."""
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
    divisors = [1]
    for p, exp in factors.items():
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(exp + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors

def _find_solution_for_n(n, deadline):
    """Find distinct x < y < z with 1/x + 1/y + 1/z = 4/n."""
    MAX_DENOM = 10**12
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        a = 4 * x - n
        b = n * x
        if a <= 0:
            continue
        
        # (a*y - b)(a*z - b) = b^2
        # Let u = a*y - b, v = a*z - b, u*v = b^2
        # y = (u + b) / a, z = (v + b) / a
        # Need: u < v, (u+b) % a == 0, (v+b) % a == 0
        #       y > x, z > y, all <= MAX_DENOM
        
        b2 = b * b
        # Factor b^2
        fb = _factorize(b)
        fb2 = {}
        for p, e in fb.items():
            fb2[p] = e * 2
        divs = _divisors_from_factors(fb2)
        
        # We need u < v = b^2/u, so u < b (since u*v = b^2 and u < v implies u^2 < b^2)
        # Also need y = (u+b)/a > x => u + b > a*x => u > a*x - b
        # And z = (v+b)/a > y => v + b > u + b => v > u (already ensured)
        # And z <= MAX_DENOM => (v+b)/a <= MAX_DENOM => v <= a*MAX_DENOM - b
        
        # Sort divisors for efficient search
        divs.sort()
        
        # Binary search for u values: u < b, u > a*x - b, (u+b) % a == 0
        # Also v = b^2/u, (v+b) % a == 0, v > u, z = (v+b)//a <= MAX_DENOM, y = (u+b)//a > x
        
        for u in divs:
            if u >= b:
                break
            if (u + b) % a != 0:
                continue
            y = (u + b) // a
            if y <= x:
                continue
            if y > MAX_DENOM:
                continue
            v = b2 // u
            if v <= u:
                continue
            if (v + b) % a != 0:
                continue
            z = (v + b) // a
            if z <= y:
                continue
            if z > MAX_DENOM:
                continue
            # Verify: 1/x + 1/y + 1/z == 4/n
            # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
    return None

def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 85  # safety margin before 90s kill
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        answers[idx] = _find_solution_for_n(n, deadline)
    
    return answers
# EVOLVE-BLOCK-END
