"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    TARGET = Fraction(5, 31)
    MAX_DEN = 10_000_000
    MAX_TERMS = 256

    # Structural approach: fix the largest denominator c and check if
    # 5/31 - 1/c can be decomposed into 1/a + 1/b with a,b < c, distinct.
    # This directly searches for the minimum c.
    
    # First: prove no 2-term solution exists.
    # 1/a + 1/b = 5/31 => (5a-31)(5b-31) = 961
    # 961 = 31^2, divisors: 1, 31, 961
    # 5a-31=1 => a=32/5 not int; 5a-31=31 => a=62/5 not int; 5a-31=961 => a=992/5 not int
    # So minimum is 3 terms.
    
    # Search for 3-term solutions by iterating c from lower bound upward.
    # For 1/a + 1/b + 1/c = 5/31 with a <= b <= c:
    #   1/a + 1/b = 5/31 - 1/c = (5c - 31)/(31c)
    # Let r = 5c - 31, s = 31c. We need 1/a + 1/b = r/s.
    # => s(a+b) = r*ab => rab - sa - sb = 0 => (ra - s)(rb - s) = s^2
    # So for each divisor d of s^2: a = (d+s)/r, b = (s^2/d + s)/r
    # Need a,b positive integers, a <= b < c, all distinct.
    
    # Lower bound for c: since 1/a + 1/b + 1/c = 5/31 and a<=b<=c,
    # 3/c >= 5/31 => c <= 93/5 = 18.6... wait that's upper bound on c from below.
    # Actually: 1/a >= 1/c, so 3*(1/c) >= 5/31 => c <= 93/5. No wait.
    # 1/a + 1/b + 1/c = 5/31, with a <= b <= c means 1/a >= 1/b >= 1/c.
    # So 3*(1/c) <= 5/31 => c >= 93/5 = 18.6, so c >= 19.
    # Also 1/a = 5/31 - 1/b - 1/c >= 5/31 - 2/c. For this to be positive: c > 62/5 = 12.4.
    # More tightly: 1/a <= 5/31 - 1/c (since 1/b > 0), so a >= 31/(5-31/c) = 31c/(5c-31).
    # For a to be integer and a <= c: 31c/(5c-31) <= c => 31 <= 5c-31 => c >= 12.4.
    
    # We'll iterate c from 19 upward and check each one.
    # Stop when we find a valid solution (that's the minimum c).
    
    best_sol = None
    
    for c in range(19, 250):  # We know 248 works, so check up to 248
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # r = 5c - 31, s = 31c
        r = 5 * c - 31
        s = 31 * c
        
        if r <= 0:
            continue
        
        # Need (ra - s)(rb - s) = s^2
        # Let d = ra - s, then a = (d + s) / r, and rb - s = s^2 / d
        # So d must divide s^2, and (d+s) must be divisible by r.
        # Also a >= 1, b >= a, b < c (since c is the largest), all distinct.
        
        s2 = s * s
        
        # d must be positive: ra > s => a > s/r = 31c/(5c-31)
        # Also d <= s^2 (since the other factor is >= 1)
        # And a = (d+s)/r <= c => d <= rc - s = c(5c-31) - 31c = 5c^2 - 62c = c(5c-62)
        # For c=19: 19*(95-62)=19*33=627
        # For c=248: 248*(1240-62)=248*1178=292144
        
        # Also b = (s^2/d + s)/r >= a and b < c
        # b < c => (s^2/d + s) < rc => s^2/d < rc - s => d > s^2/(rc - s)
        
        max_d = c * (5 * c - 62)
        if max_d <= 0:
            continue
        
        # d must divide s^2, d in range (s^2/(rc-s), max_d], and (d+s) % r == 0
        # Let's find valid d values.
        
        found = False
        # d must satisfy: d | s^2, (d+s) % r == 0, d > s^2/(rc-s), d <= max_d
        # (d+s) % r == 0 means d ≡ -s (mod r)
        # s = 31c, r = 5c-31
        # s mod r: 31c mod (5c-31)
        # Let's just iterate over divisors of s^2 in the valid range.
        
        # s^2 = 961 * c^2. For c up to 248, c^2 up to ~61504, s^2 up to ~59M.
        # Number of divisors of s^2 is manageable.
        
        # More efficient: d = k*r - s for integer k, and d | s^2.
        # d > 0 => k*r > s => k > s/r = 31c/(5c-31)
        # d <= max_d => k*r - s <= max_d => k <= (max_d + s) / r
        
        k_min = s // r + 1
        k_max = (max_d + s) // r
        
        for k in range(k_min, k_max + 1):
            d = k * r - s
            if d <= 0:
                continue
            if d > max_d:
                break
            if s2 % d != 0:
                continue
            
            other = s2 // d
            # a = (d + s) / r, b = (other + s) / r
            if (d + s) % r != 0:
                continue
            if (other + s) % r != 0:
                continue
            
            a = (d + s) // r
            b = (other + s) // r
            
            if a < 1 or b < 1:
                continue
            if a > b:
                a, b = b, a
            # Check: a <= b < c, all distinct
            if b >= c:
                continue
            if a == b or a == c or b == c:
                continue
            if a < 1:
                continue
            
            # Verify: 1/a + 1/b + 1/c == 5/31
            check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if check == TARGET:
                best_sol = [a, b, c]
                found = True
                break
        
        if found:
            break
    
    if best_sol is None:
        # Fallback to known solution
        best_sol = [8, 31, 248]
    
    # Final verification
    s = sum(Fraction(1, d) for d in best_sol)
    if s != TARGET or len(set(best_sol)) != len(best_sol) or any(d < 1 or d > MAX_DEN for d in best_sol) or len(best_sol) > MAX_TERMS:
        best_sol = [8, 31, 248]
    
    return best_sol
# EVOLVE-BLOCK-END
