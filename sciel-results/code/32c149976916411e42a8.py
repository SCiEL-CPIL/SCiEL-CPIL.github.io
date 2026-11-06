"""Rewritten program with structured search for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    
    def valid_solution(denoms):
        """Check if denominators form a valid solution."""
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target
    
    def find_2_terms(rem, max_d, excl):
        """Find 2 distinct denominators x, y > x, both <= max_d, not in excl, with 1/x + 1/y = rem."""
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        # 1/x + 1/y = p/q => (x+y)/(xy) = p/q => q(x+y) = pxy
        # => pxy - qx - qy = 0 => (px-q)(py-q) = q^2
        # For given x: y = qx / (px - q), need px - q > 0 and divides qx
        lo = q // p + 1
        if lo < 1:
            lo = 1
        hi = max_d
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            k = p * x - q
            if k <= 0:
                continue
            if (q * x) % k != 0:
                continue
            y = (q * x) // k
            if y <= x or y > max_d or y in excl:
                continue
            return (x, y)
        return None
    
    def find_3_terms(rem, max_d, excl):
        """Find 3 distinct denominators x < y < z, all <= max_d, not in excl, with 1/x+1/y+1/z = rem."""
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        # Upper bound for x: 1/x < rem, and we need 2 more terms, so 1/x < rem - 1/max_d - 1/max_d
        # But simpler: x < q/p, and also 3/x > rem so x < 3q/p
        hi = min(3 * q // p + 1, max_d)
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            r2 = rem - Fraction(1, x)
            if r2 <= 0:
                break
            r = find_2_terms(r2, max_d, excl | {x})
            if r is not None:
                return (x, r[0], r[1])
        return None
    
    def find_4_terms(rem, max_d, excl):
        """Find 4 distinct denominators, all <= max_d, not in excl."""
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        hi = min(4 * q // p + 1, max_d)
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            r2 = rem - Fraction(1, x)
            if r2 <= 0:
                break
            r = find_3_terms(r2, max_d, excl | {x})
            if r is not None:
                return (x,) + r
        return None
    
    def find_5_terms(rem, max_d, excl):
        """Find 5 distinct denominators, all <= max_d, not in excl."""
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        hi = min(5 * q // p + 1, max_d)
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            if time.time() > deadline:
                return None
            r2 = rem - Fraction(1, x)
            if r2 <= 0:
                break
            r = find_4_terms(r2, max_d, excl | {x})
            if r is not None:
                return (x,) + r
        return None
    
    def find_6_terms(rem, max_d, excl):
        """Find 6 distinct denominators, all <= max_d, not in excl."""
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        hi = min(6 * q // p + 1, max_d)
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            if time.time() > deadline:
                return None
            r2 = rem - Fraction(1, x)
            if r2 <= 0:
                break
            r = find_5_terms(r2, max_d, excl | {x})
            if r is not None:
                return (x,) + r
        return None
    
    def can_solve(rem, max_d, excl, max_terms):
        """Try to find up to max_terms denominators summing to rem, all <= max_d."""
        if max_terms <= 0:
            return [] if rem == 0 else None
        if max_terms == 1:
            if rem.denominator == 1:
                x = rem.numerator
                if 1 <= x <= max_d and x not in excl:
                    return [x]
            return None
        if max_terms == 2:
            r = find_2_terms(rem, max_d, excl)
            return list(r) if r else None
        if max_terms == 3:
            r = find_3_terms(rem, max_d, excl)
            return list(r) if r else None
        if max_terms == 4:
            r = find_4_terms(rem, max_d, excl)
            return list(r) if r else None
        if max_terms == 5:
            r = find_5_terms(rem, max_d, excl)
            return list(r) if r else None
        if max_terms == 6:
            r = find_6_terms(rem, max_d, excl)
            return list(r) if r else None
        # For more terms, use recursive approach with time limit
        p, q = rem.numerator, rem.denominator
        if p <= 0:
            return None
        lo = q // p + 1
        if lo < 1:
            lo = 1
        hi = min(max_terms * q // p + 1, max_d)
        for x in range(lo, hi + 1):
            if x in excl:
                continue
            if time.time() > deadline:
                return None
            r2 = rem - Fraction(1, x)
            if r2 <= 0:
                break
            r = can_solve(r2, max_d, excl | {x}, max_terms - 1)
            if r is not None:
                return [x] + r
        return None
    
    def check_max_denom(C):
        """Check if there's a solution with all denominators <= C."""
        if time.time() > deadline:
            return None
        
        # Try 1 term
        if target.denominator == 1:
            x = target.numerator
            if 1 <= x <= C:
                return [x]
        
        # Try 2 terms: 1/x + 1/y = 5/31, x < y <= C
        r = find_2_terms(target, C, frozenset())
        if r is not None:
            return sorted([r[0], r[1]])
        
        # Try 3 terms
        r = find_3_terms(target, C, frozenset())
        if r is not None:
            return sorted([r[0], r[1], r[2]])
        
        # Try 4 terms
        r = find_4_terms(target, C, frozenset())
        if r is not None:
            return sorted(list(r))
        
        # Try 5 terms
        r = find_5_terms(target, C, frozenset())
        if r is not None:
            return sorted(list(r))
        
        # Try 6 terms
        r = find_6_terms(target, C, frozenset())
        if r is not None:
            return sorted(list(r))
        
        # Try up to 8 terms with more general search
        for nt in range(7, 9):
            if time.time() > deadline:
                break
            r = can_solve(target, C, frozenset(), nt)
            if r is not None:
                return sorted(r)
        
        return None
    
    # Binary search on maximum denominator
    # Lower bound: need at least 1/x = 5/31, so x >= 31/5 = 6.2, so x >= 7
    # But with multiple terms, the max could be smaller
    # Upper bound: known solution has max 248
    lo = 7
    hi = 248
    best_solution = None
    
    # First, verify the known solution
    known = [8, 31, 248]
    if valid_solution(known):
        best_solution = known
    
    # Binary search for the minimum C
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = check_max_denom(mid)
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Also try a direct search from low to find if we missed anything
    if time.time() < deadline:
        for C in range(7, min(lo, 250)):
            if time.time() > deadline:
                break
            sol = check_max_denom(C)
            if sol is not None:
                best_solution = sol
                break
    
    if best_solution is None:
        return [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
