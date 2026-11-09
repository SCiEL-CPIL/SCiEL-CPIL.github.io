import time
from fractions import Fraction
from math import isqrt

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _get_divisors(n):
    """Get all divisors of n."""
    if n == 0:
        return []
    divs = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

def _find_solution_for_max_fast(M):
    """
    Try to find a set of distinct denominators in [1, M] that sum to TARGET.
    Uses divisor-based enumeration for 3 and 4 term solutions.
    Returns a list of denominators if found, else None.
    """
    # 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c <= M
    # a must be such that 1/a < 5/31 => a > 6.2 => a >= 7.
    # And 3/a > 5/31 => a < 18.6 => a <= 18.
    # But if we want small M, a can be larger?
    # If a > 18, 1/a < 5/31/3.
    # 1/b + 1/c > 2/3 * 5/31? No.
    # If a is large, 1/b + 1/c is close to 5/31.
    # b, c must be around 31/5 = 6.2?
    # But b > a. So if a > 18, b > 18.
    # 1/b + 1/c < 2/18 = 1/9 = 0.111.
    # 5/31 = 0.161.
    # So 1/b + 1/c < 5/31 - 1/a.
    # If a=19, 1/b + 1/c = 5/31 - 1/19 = (95-31)/589 = 64/589 = 0.108.
    # Min 1/b + 1/c with b>19 is 1/20 + 1/21 = 0.097.
    # So it's possible.
    
    # General bound for a:
    # 1/a + 1/(a+1) + 1/(a+2) >= 5/31?
    # If not, a can be larger.
    # But we iterate a from 7 upwards.
    # Upper bound for a:
    # 3/a > 5/31 is not required if we allow b,c to be close to a.
    # But b > a, c > b.
    # So 1/a + 1/b + 1/c < 3/a.
    # We need 3/a >= 5/31 => a <= 93/5 = 18.6.
    # So a must be <= 18 for a 3-term solution with distinct a<b<c?
    # Wait. 1/a + 1/b + 1/c = 5/31.
    # Since b > a and c > b, 1/b < 1/a and 1/c < 1/a.
    # So 5/31 = 1/a + 1/b + 1/c < 3/a.
    # So 3/a > 5/31 => a < 93/5 = 18.6.
    # So a is in [7, 18].
    
    for a in range(7, min(19, M) + 1):
        N = 5 * a - 31
        D = 31 * a
        if N <= 0:
            continue
        
        # We need (Nb - D)(Nc - D) = D^2
        # Let X = Nb - D, Y = Nc - D.
        # X * Y = D^2.
        # b = (X + D) / N, c = (Y + D) / N.
        # b, c must be integers.
        # X + D must be divisible by N.
        # X + D = Nb => X = Nb - D.
        # X mod N = -D mod N.
        # So we need divisors X of D^2 such that X % N == (-D) % N.
        
        target_mod = (-D) % N
        divs = _get_divisors(D * D)
        
        found = False
        for X in divs:
            if X % N != target_mod:
                continue
            Y = (D * D) // X
            if Y % N != target_mod:
                continue
            
            b = (X + D) // N
            c = (Y + D) // N
            
            # Check distinctness and bounds
            if b > a and c > b and c <= M:
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                    return [a, b, c]
            
            # Also check swapped b, c? 
            # Since we iterate all divisors, we will hit both X and Y.
            # But we require b < c.
            # If X < Y, then b < c.
            # If X > Y, then b > c.
            # We just need to check if the set {b,c} works.
            # The check c > b handles the ordering.
            # If the solution has b > c, we would have found it when X was the larger divisor?
            # No, if X is larger, b is larger.
            # We require b > a and c > b.
            # So we specifically look for b < c.
            # If the solution is a, c, b (with c < b), we would miss it if we only check c > b.
            # But the set is unordered in the problem statement? "1..256 distinct denominators".
            # The order doesn't matter for the sum.
            # So we can just check if {b, c} are valid distinct denominators > a.
            # And min(b,c) > a.
            
            if b != c and b > a and c > a:
                min_bc = min(b, c)
                max_bc = max(b, c)
                if min_bc > a and max_bc <= M:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                        return [a, b, c]
        
    # 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d <= M
    # a <= 13? 
    # 4/a > 5/31 => a < 124/5 = 24.8.
    # So a in [7, 24].
    
    for a in range(7, min(25, M) + 1):
        rem_a = TARGET - Fraction(1, a)
        if rem_a <= 0:
            continue
        
        # 1/b + 1/c + 1/d = rem_a
        # b > a.
        # 3/b > rem_a => b < 3/rem_a.
        # Also 1/b < rem_a => b > 1/rem_a.
        
        b_max = min(M, int(3 / rem_a))
        # b must be > a.
        for b in range(a + 1, b_max + 1):
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            
            # 1/c + 1/d = rem_b
            # c > b.
            # N_b = rem_b.numerator, D_b = rem_b.denominator.
            N_b = rem_b.numerator
            D_b = rem_b.denominator
            
            # (N_b c - D_b)(N_b d - D_b) = D_b^2
            # X = N_b c - D_b.
            # X % N_b == -D_b % N_b.
            
            target_mod = (-D_b) % N_b
            divs = _get_divisors(D_b * D_b)
            
            for X in divs:
                if X % N_b != target_mod:
                    continue
                Y = (D_b * D_b) // X
                if Y % N_b != target_mod:
                    continue
                
                c = (X + D_b) // N_b
                d = (Y + D_b) // N_b
                
                if c != d and c > b and d > b:
                    min_cd = min(c, d)
                    max_cd = max(c, d)
                    if min_cd > b and max_cd <= M:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == TARGET:
                            return [a, b, c, d]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    known = [8, 31, 248]
    best = known
    
    # Search for smaller max denominators
    for M in range(247, 1, -1):
        if time.time() > deadline:
            break
        sol = _find_solution_for_max_fast(M)
        if sol is not None:
            return sol
            
    return best
