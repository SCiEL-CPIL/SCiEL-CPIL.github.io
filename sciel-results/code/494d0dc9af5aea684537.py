"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

def _factorize(n):
    """Return dict of prime->exponent for n."""
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

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    # Precompute factorizations for small numbers
    max_b = 256 * 192 + 100
    factor_cache = {}
    for i in range(2, max_b + 1):
        factor_cache[i] = _factorize(i)
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            b_sq = b * b
            
            min_U = 2 * x * (2 * x - n) + 1
            if min_U < 1:
                min_U = 1
            
            found_for_x = None
            
            # Estimate cost of both approaches
            # Approach 1: divisor generation - cost ~ number of divisors of b_sq
            # Approach 2: k iteration - cost ~ k_range size
            k_lo = (min_U + b + a - 1) // a
            k_hi = (2 * b - 1) // a
            k_range_size = max(0, k_hi - k_lo + 1)
            
            # Get divisor count estimate
            b_factors = factor_cache.get(b)
            if b_factors is None:
                b_factors = _factorize(b)
            
            # Number of divisors of b_sq = product of (2*e_i + 1)
            div_count = 1
            for p, e in b_factors.items():
                div_count *= (2 * e + 1)
            
            # Choose approach: use divisors if div_count < k_range_size, else use k iteration
            if div_count < k_range_size:
                # Divisor generation approach
                b_sq_factors = {}
                for p, e in b_factors.items():
                    b_sq_factors[p] = e * 2
                
                divs = _divisors_from_factors(b_sq_factors)
                
                for U in divs:
                    if U < min_U:
                        continue
                    V = b_sq // U
                    if V < min_U:
                        continue
                    if (U + b) % a != 0:
                        continue
                    if (V + b) % a != 0:
                        continue
                    y = (U + b) // a
                    z = (V + b) // a
                    if y > z:
                        y, z = z, y
                    if y <= x or z <= y:
                        continue
                    if y > BOUND or z > BOUND:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    found_for_x = [x, y, z]
                    break
            else:
                # k iteration approach
                for k in range(k_lo, k_hi + 1):
                    U = k * a - b
                    if U <= 0:
                        continue
                    if b_sq % U != 0:
                        continue
                    V = b_sq // U
                    if (U + b) % a != 0:
                        continue
                    if (V + b) % a != 0:
                        continue
                    y = (U + b) // a
                    z = (V + b) // a
                    if y > z:
                        y, z = z, y
                    if y <= x or z <= y:
                        continue
                    if y > BOUND or z > BOUND:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    found_for_x = [x, y, z]
                    break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
