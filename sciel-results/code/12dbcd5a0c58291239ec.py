import time
from math import isqrt

def _factorize(n):
    """Return list of (prime, exponent) pairs for n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        current = []
        pe = 1
        for i in range(e + 1):
            for d in divisors:
                current.append(d * pe)
            pe *= p
        divisors = current
    return divisors

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_DENOM = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # Try small k values: x = n//4 + k (and nearby)
        # 4x - n = 4k (when x = n/4 + k exactly, but n may not be divisible by 4)
        # More generally, try x from floor(n/4)+1 to some bound
        x_min = n // 4 + 1
        
        # For efficiency, try x values where 4x-n is small
        # x = ceil(n/4) + k for k = 0, 1, 2, ...
        # But we need x > n/4, so x >= floor(n/4) + 1
        
        # Try a range of x values, prioritizing those where 4x-n is small
        # 4x - n = a, so x = (n + a) / 4, a must have same parity as n
        # For small a, x is close to n/4
        
        # Strategy: try x values in order of |4x - n|
        candidates = []
        # x must satisfy: n/4 < x <= 3n/4 (for x <= y <= z)
        x_low = n // 4 + 1
        x_high = (3 * n) // 4
        
        # Generate candidates sorted by |4x - n|
        for x in range(x_low, x_high + 1):
            a = 4 * x - n  # positive
            candidates.append((a, x))
        
        # Sort by a (smaller a = easier factorization)
        candidates.sort()
        
        for a, x in candidates:
            if time.time() > deadline:
                break
            
            b = n * x  # so a*y*z = b*(y+z)
            # (a*y - b)(a*z - b) = b^2
            b_sq = b * b
            
            # Factor b_sq
            factors = _factorize(b_sq)
            divs = _divisors_from_factors(factors)
            
            for s in divs:
                if s * s > b_sq:
                    break
                if b_sq % s != 0:
                    continue
                t = b_sq // s
                
                # y = (s + b) / a, z = (t + b) / a
                if (s + b) % a != 0 or (t + b) % a != 0:
                    continue
                y = (s + b) // a
                z = (t + b) // a
                
                # Check distinctness and bounds
                if y <= 0 or z <= 0:
                    continue
                if x == y or x == z or y == z:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                if x > MAX_DENOM:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
