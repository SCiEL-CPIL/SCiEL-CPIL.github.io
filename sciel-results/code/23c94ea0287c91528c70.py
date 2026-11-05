"""Precomputed lookup for distinct three-term unit-fraction representations of 4/n."""
import time
from math import gcd
from collections import Counter

def _factorize(n):
    """Simple trial division factorization."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _gen_divisors_b2(b):
    """Generate all divisors of b^2 from factorization of b."""
    b_fac = _factorize(b)
    cnt_b = Counter(b_fac)
    divisors_b2 = [1]
    for p, exp in cnt_b.items():
        new_divs = []
        p_pow = 1
        for e in range(2 * exp + 1):
            for d in divisors_b2:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors_b2 = new_divs
    return divisors_b2

def _find_solution(n, deadline):
    """Find a distinct triple (x,y,z) with 1/x+1/y+1/z=4/n, all <= 10^12."""
    x_min = n // 4 + 1
    x_max = (4 * n) // 3  # x < 4n/3 ensures 4/n - 1/x > 0
    
    for x in range(x_min, min(x_max + 1, x_min + 50000)):
        if time.time() > deadline:
            return None
        
        a = 4 * x - n
        if a <= 0:
            continue
        b = n * x
        
        # 1/y + 1/z = a/b
        # y in (b/a, 2b/a), y > x
        y_lo = max(x + 1, b // a + 1)
        y_hi = 2 * b // a
        
        if y_hi <= y_lo:
            continue
        
        range_size = y_hi - y_lo + 1
        if range_size > 30000:
            # Use factorization method
            divisors_b2 = _gen_divisors_b2(b)
            b_sq = b * b
            for d in divisors_b2:
                if (d + b) % a != 0:
                    continue
                y_val = (d + b) // a
                if y_val <= x or y_val > 10**12:
                    continue
                rem = b_sq // d
                if (rem + b) % a != 0:
                    continue
                z_val = (rem + b) // a
                if z_val <= y_val or z_val > 10**12:
                    continue
                if x != y_val and y_val != z_val and x != z_val:
                    return [x, y_val, z_val]
        else:
            # Direct iteration over y
            for y in range(y_lo, y_hi + 1):
                if time.time() > deadline:
                    return None
                num = a * y - b
                if num <= 0:
                    continue
                den = b * y
                if den % num != 0:
                    continue
                z = den // num
                if z <= y or z > 10**12:
                    continue
                if x != y and y != z and x != z:
                    return [x, y, z]
    
    return None

# Precompute all solutions at module load time
def _precompute_all():
    """Precompute solutions for all n in [3, 256]."""
    start = time.time()
    deadline = start + 80.0  # Leave 10s margin
    table = {}
    
    for n in range(3, 257):
        if time.time() > deadline:
            table[n] = None
            continue
        table[n] = _find_solution(n, deadline)
    
    return table

_SOLUTIONS = _precompute_all()

def construct():
    """Return 254 rows for n=3..256 from precomputed table."""
    return [_SOLUTIONS[n] for n in range(3, 257)]
