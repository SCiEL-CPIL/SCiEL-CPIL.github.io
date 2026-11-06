"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    
    # Theoretical lower bound for the largest denominator C:
    # If we have k terms, the sum is at most k/C (if all were C, but they are distinct and smaller).
    # Actually, the largest term 1/C must be such that the sum of the rest can reach the remainder.
    # A better lower bound: 1/C < 5/31 => C > 31/5 = 6.2, so C >= 7.
    # But we need distinct denominators.
    # Let's search for C starting from a reasonable lower bound.
    # For 3 terms: 1/a + 1/b + 1/c = 5/31 with a < b < c.
    # c must be > 31/5 = 6.2.
    # Let's try to find a solution with small C.
    
    def can_solve_2_terms(rem, max_denom, exclude):
        """
        Check if rem can be written as 1/x + 1/y with x < y <= max_denom,
        x, y distinct and not in exclude.
        rem = p/q.
        1/x + 1/y = p/q => (px-q)(py-q) = q^2.
        """
        p = rem.numerator
        q = rem.denominator
        if p == 0:
            return []
        if p < 0:
            return []
            
        # (px-q)(py-q) = q^2
        # Let u = px - q, v = py - q.
        # u*v = q^2.
        # x = (u+q)/p, y = (v+q)/p.
        # We need x < y, so u < v, so u < q.
        # Also x < y => u < v.
        # x, y must be integers, so (u+q) % p == 0 and (v+q) % p == 0.
        # x <= max_denom, y <= max_denom.
        # x, y not in exclude.
        
        # Iterate divisors u of q^2 such that u < q
        # This is expensive if q is large.
        # Alternative: Iterate x from 1 to max_denom.
        # 1/y = rem - 1/x = (p*x - q) / (q*x)
        # y = q*x / (p*x - q)
        # For y to be integer, (p*x - q) must divide q*x.
        # Let g = gcd(p*x - q, q). Then p*x - q must divide q/g? No.
        # Let k = p*x - q. k > 0 => x > q/p.
        # y = q*x / k.
        # k must divide q*x.
        
        # Since max_denom is small in our search (we start small), iterating x is fast.
        min_x = q // p + 1  # x must be > q/p for k > 0
        if min_x < 1:
            min_x = 1
            
        solutions = []
        for x in range(min_x, max_denom + 1):
            if x in exclude:
                continue
            k = p * x - q
            if k <= 0:
                continue
            if (q * x) % k != 0:
                continue
            y = (q * x) // k
            if y <= x:
                continue
            if y > max_denom:
                continue
            if y in exclude:
                continue
            solutions.append((x, y))
            # Since we want distinct and x < y, and we iterate x increasing, 
            # we might find multiple, but we just need one valid set.
            break 
        return solutions

    def can_solve_3_terms(rem, max_denom, exclude):
        """
        Check if rem can be written as 1/x + 1/y + 1/z with x < y < z <= max_denom,
        distinct and not in exclude.
        """
        p = rem.numerator
        q = rem.denominator
        if p <= 0:
            return []
        
        min_x = q // p + 1
        if min_x < 1:
            min_x = 1
            
        # x < y < z
        # 1/x < rem => x > q/p
        # 1/x + 1/y > rem - 1/max_denom (approx)
        # 1/x < rem => x > q/p
        # Also 3/x > rem => x < 3q/p
        
        max_x = min(3 * q // p, max_denom)
        
        for x in range(min_x, max_x + 1):
            if x in exclude:
                continue
            rem2 = rem - Fraction(1, x)
            if rem2 <= 0:
                break # x too small, 1/x > rem
            # Now solve 1/y + 1/z = rem2 with y > x, z > y, z <= max_denom
            sols = can_solve_2_terms(rem2, max_denom, exclude.union([x]))
            if sols:
                y, z = sols[0]
                if y > x and z > y:
                    return [x, y, z]
        return []

    def can_solve_4_terms(rem, max_denom, exclude):
        """
        Check if rem can be written as 1/x + 1/y + 1/z + 1/w with x < y < z < w <= max_denom.
        """
        p = rem.numerator
        q = rem.denominator
        if p <= 0:
            return []
        
        min_x = q // p + 1
        max_x = min(4 * q // p, max_denom)
        
        for x in range(min_x, max_x + 1):
            if x in exclude:
                continue
            rem2 = rem - Fraction(1, x)
            if rem2 <= 0:
                break
            sols = can_solve_3_terms(rem2, max_denom, exclude.union([x]))
            if sols:
                y, z, w = sols
                if y > x and z > y and w > z:
                    return [x, y, z, w]
        return []

    # Search for the minimum largest denominator C.
    # We know C must be at least 7.
    # Let's try C from 7 upwards.
    # For a fixed C, the remaining sum is R = 5/31 - 1/C.
    # We need to represent R as a sum of distinct unit fractions 1/d_i with d_i < C.
    # The number of terms is not fixed, but to minimize C, we likely need few terms.
    # Let's check for 1, 2, 3, 4 terms.
    
    for C in range(7, 1000):
        if time.time() > deadline:
            break
            
        rem = target - Fraction(1, C)
        if rem <= 0:
            continue
            
        # Try 1 term: rem = 1/d => d = 1/rem. Must be integer and d < C.
        if rem.denominator == 1:
            d = rem.numerator
            if d < C and d != C:
                return [C, d] # Order doesn't matter for the set, but let's return sorted
                # Wait, the problem asks for distinct denominators. [C, d] is valid if d != C.
                # Actually, we just need to return the list.
                return sorted([C, d])
                
        # Try 2 terms: rem = 1/d1 + 1/d2 with d1 < d2 < C.
        sols = can_solve_2_terms(rem, C - 1, set())
        if sols:
            d1, d2 = sols[0]
            return sorted([C, d1, d2])
            
        # Try 3 terms: rem = 1/d1 + 1/d2 + 1/d3 with d1 < d2 < d3 < C.
        sols = can_solve_3_terms(rem, C - 1, set())
        if sols:
            d1, d2, d3 = sols
            return sorted([C, d1, d2, d3])
            
        # Try 4 terms: rem = 1/d1 + 1/d2 + 1/d3 + 1/d4 with d1 < d2 < d3 < d4 < C.
        sols = can_solve_4_terms(rem, C - 1, set())
        if sols:
            d1, d2, d3, d4 = sols
            return sorted([C, d1, d2, d3, d4])

    # Fallback if no small C found (unlikely given the problem nature)
    # Use the original greedy method as a last resort
    r = target
    values = []
    while r > 0:
        if time.time() > deadline:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in values:
            # Greedy might pick duplicates, need to adjust?
            # For now, just append, but the task requires distinct.
            # A simple greedy might fail distinctness.
            # Let's try a slightly smarter greedy or just return what we have if valid.
            pass
        values.append(d)
        r -= Fraction(1, d)
        if len(values) > 256:
            break
            
    # Ensure distinctness for fallback? 
    # If greedy fails distinctness, it's invalid.
    # But for 5/31, greedy is:
    # 5/31 -> 1/7 (rem 18/217)
    # 18/217 -> 1/13 (rem 11/2821)
    # 11/2821 -> 1/257 (rem 1/724997)
    # 1/724997 -> 1/724997
    # Denoms: 7, 13, 257, 724997. All distinct. Max is 724997.
    # This is a valid solution, just not optimal.
    
    return values
# EVOLVE-BLOCK-END
