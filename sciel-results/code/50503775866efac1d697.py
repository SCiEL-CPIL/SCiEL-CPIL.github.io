"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution_for_n(n, deadline):
    """Find distinct positive integer denominators x,y,z with 1/x+1/y+1/z=4/n."""
    # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct)
    # For a given x: A = 4x - n, B = n*x
    # Need (A*y - B)(A*z - B) = B^2
    # So d = A*y - B must be a divisor of B^2, d < B (since y < z)
    # y = (d + B) / A must be integer, y > x
    # z = (B^2/d + B) / A must be integer, z > y
    
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        
        # y_min: need A*y - B > 0 => y > B/A, and y > x
        y_min = max(x + 1, B // A + 1)
        # y_max: need z > y => y < 2B/A
        y_max = (2 * B) // A
        
        if y_max < y_min:
            continue
        
        # Factor B^2 to enumerate divisors
        # B = n * x, factor B then double exponents for B^2
        fb = _factorize(B)
        fb2 = {p: 2*e for p, e in fb.items()}
        
        # We need divisors d of B^2 such that:
        #   d < B (so that y < z)
        #   (d + B) % A == 0 (so y is integer)
        #   y = (d+B)//A > x
        #   (B^2//d + B) % A == 0 (so z is integer)
        #   z = (B^2//d + B)//A > y
        #   z <= 10^12
        
        B2 = B * B
        divs = _divisors_from_factors(fb2)
        
        # Sort divisors for early termination potential
        # We need d < B, and (d+B) divisible by A
        for d in divs:
            if time.time() > deadline:
                return None
            if d >= B:
                continue
            # y = (d + B) / A
            num_y = d + B
            if num_y % A != 0:
                continue
            y = num_y // A
            if y <= x:
                continue
            if y < y_min or y > y_max:
                continue
            # z = (B^2/d + B) / A
            num_z = B2 // d + B
            if num_z % A != 0:
                continue
            z = num_z // A
            if z <= y:
                continue
            if z > 10**12:
                continue
            # Verify: 1/x + 1/y + 1/z == 4/n
            # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, deadline)
    
    return results
# EVOLVE-BLOCK-END
