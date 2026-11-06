"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _get_divisors(n):
    """Get all divisors of n."""
    if n == 0:
        return []
    divisors = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
        i += 1
    return sorted(divisors)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # For given a, r1 = 5/31 - 1/a = (5a-31)/(31a)
    # Need 1/b + 1/c = r1 = p/q where p = 5a-31, q = 31a
    # Then (pb - q)(pc - q) = q^2
    # So we need divisors d of q^2 such that d > q and q^2/d > q
    # and both (d+q)/p and (q^2/d + q)/p are positive integers > a
    
    # a must satisfy: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 1/a + 1/(a+1) + 1/(a+2) > 5/31 for small a
    # Let's search a from 7 upward
    
    for a in range(7, 200):
        if time.time() > deadline:
            break
        
        p = 5 * a - 31
        q = 31 * a
        
        if p <= 0:
            continue
        
        # (pb - q)(pc - q) = q^2
        # Let d = pb - q, then d must be a divisor of q^2
        # d > 0, and b = (d + q) / p must be integer
        # c = (q^2/d + q) / p must be integer
        # We want a < b <= c (for distinctness and ordering)
        
        q2 = q * q
        
        # Get divisors of q^2 that are > q (so that b > a roughly)
        # Actually we need b > a, so pb - q > pa - q => d > pa - q
        # pa - q = p*a - q = (5a-31)*a - 31a = 5a^2 - 31a - 31a = 5a^2 - 62a
        
        min_d = p * a - q + 1  # d > pa - q for b > a
        
        # Also need c >= b, so q^2/d <= d => d >= q (approximately)
        # More precisely: c >= b means (q^2/d + q)/p >= (d + q)/p
        # => q^2/d + q >= d + q
        # => q^2/d >= d
        # => d^2 <= q^2
        # => d <= q
        
        # Wait, that's contradictory. Let me re-examine.
        # If d = pb - q and e = pc - q, then d*e = q^2
        # b = (d+q)/p, c = (e+q)/p
        # For b <= c, we need d <= e, so d <= q (since d*e = q^2 and d,e > 0)
        
        # So we need d such that:
        # 1. d divides q^2
        # 2. d <= q (for b <= c)
        # 3. d > pa - q (for b > a)
        # 4. (d + q) % p == 0 (for b integer)
        # 5. (q^2//d + q) % p == 0 (for c integer)
        # 6. b != c (distinctness) - but if b == c, we skip
        
        divs = _get_divisors(q2)
        
        for d in divs:
            if d > q:
                break  # divisors are sorted, so we can stop
            if d < min_d:
                continue
            
            # Check if (d + q) is divisible by p
            if (d + q) % p != 0:
                continue
            
            b = (d + q) // p
            
            # Check if b > a
            if b <= a:
                continue
            
            # Check if (q^2//d + q) is divisible by p
            e = q2 // d
            if (e + q) % p != 0:
                continue
            
            c = (e + q) // p
            
            # Check distinctness and ordering
            if b == c:
                continue
            if b > c:
                continue  # Shouldn't happen if d <= q
            
            max_d = c
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = [a, b, c]
        
        # Early termination if we found a good solution
        if best_max_denom <= a + 20:
            break
    
    if best_solution:
        return best_solution
    
    # Fallback: Greedy
    r = target
    values = []
    while r:
        if time.time() > deadline:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
