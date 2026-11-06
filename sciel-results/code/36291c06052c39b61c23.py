"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    target = Fraction(5, 31)
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
            
    def search_4terms(p, q, min_den, max_den):
        """Find 1/b + 1/c + 1/d + 1/e = p/q with min_den < b < c < d < e < max_den."""
        results = []
        # b range
        b_lo = min_den + 1
        # 1/b < p/q => b > q/p
        b_hi = min(max_den - 3, (2 * q) // p) # Loose upper bound, refined inside
        
        for b in range(b_lo, b_hi + 1):
            rem1 = Fraction(p, q) - Fraction(1, b)
            if rem1 <= 0:
                continue
            p1, q1 = rem1.numerator, rem1.denominator
            
            # c range
            c_lo = b + 1
            # 1/c < rem1 => c > q1/p1
            c_hi = min(max_den - 2, (2 * q1) // p1)
            
            for c in range(c_lo, c_hi + 1):
                rem2 = rem1 - Fraction(1, c)
                if rem2 <= 0:
                    continue
                p2, q2 = rem2.numerator, rem2.denominator
                
                # d range
                d_lo = c + 1
                # 1/d < rem2 => d > q2/p2
                d_hi = min(max_den - 1, (2 * q2) // p2)
                
                for d in range(d_lo, d_hi + 1):
                    rem3 = rem2 - Fraction(1, d)
                    if rem3 <= 0:
                        continue
                    p3, q3 = rem3.numerator, rem3.denominator
                    
                    # e is determined
                    # 1/e = rem3 => e = q3/p3
                    if p3 == 0:
                        continue
                    if q3 % p3 != 0:
                        continue
                    e = q3 // p3
                    if e > d and e < max_den:
                        results.append((b, c, d, e))
                        
        return results

    def search_3terms(p, q, min_den, max_den):
        """Find 1/b + 1/c + 1/d = p/q with min_den < b < c < d < max_den."""
        results = []
        b_lo = min_den + 1
        b_hi = min(max_den - 2, (2 * q) // p)
        
        for b in range(b_lo, b_hi + 1):
            rem1 = Fraction(p, q) - Fraction(1, b)
            if rem1 <= 0:
                continue
            p1, q1 = rem1.numerator, rem1.denominator
            
            c_lo = b + 1
            c_hi = min(max_den - 1, (2 * q1) // p1)
            
            for c in range(c_lo, c_hi + 1):
                rem2 = rem1 - Fraction(1, c)
                if rem2 <= 0:
                    continue
                p2, q2 = rem2.numerator, rem2.denominator
                
                if p2 == 0:
                    continue
                if q2 % p2 != 0:
                    continue
                d = q2 // p2
                if d > c and d < max_den:
                    results.append((b, c, d))
                    
        return results

    def search_2terms(p, q, min_den, max_den):
        """Find 1/b + 1/c = p/q with min_den < b < c < max_den."""
        results = []
        b_lo = max(min_den + 1, q // p + 1)
        b_hi = min(max_den - 1, (2 * q) // p - 1)
        if b_hi <= b_lo:
            return results
        for b in range(b_lo, b_hi + 1):
            num = p * b - q
            if num <= 0:
                continue
            if (q * b) % num != 0:
                continue
            c = (q * b) // num
            if c > b and c < max_den:
                results.append((b, c))
        return results

    # Search for improvements
    # Try starting denominators from 7 upwards
    # 5/31 approx 0.161. 1/7 approx 0.142. 1/8 approx 0.125.
    
    for a in range(7, 31):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
            
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        p, q = rem.numerator, rem.denominator
        
        # Try 2 terms
        for b, c in search_2terms(p, q, a, best_max):
            check_and_update([a, b, c])
            
        # Try 3 terms
        for b, c, d in search_3terms(p, q, a, best_max):
            check_and_update([a, b, c, d])
            
        # Try 4 terms
        for b, c, d, e in search_4terms(p, q, a, best_max):
            check_and_update([a, b, c, d, e])
            
        # If best_max is reduced, subsequent searches will be faster/tighter
        # We could break if best_max is very small, but let's run a few more iterations if time permits

    return best
# EVOLVE-BLOCK-END
