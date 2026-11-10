"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = None
    best_result = None
    
    # Helper: try to complete a partial sum with up to k more terms,
    # all denominators > last_denom, minimizing the max denominator
    def try_complete(remaining, last_denom, max_terms, current_max):
        """Try to find up to max_terms distinct denominators > last_denom
        whose reciprocals sum to 'remaining'. Return list or None."""
        if remaining == 0:
            return []
        if max_terms == 0:
            return None
        if remaining < 0:
            return None
        
        # The next denominator must satisfy: 1/d <= remaining, so d >= ceil(1/remaining)
        # Also d > last_denom
        min_d = max(last_denom + 1, (1 // remaining.numerator) * remaining.denominator + 1 if remaining.numerator > 0 else last_denom + 1)
        # Actually: 1/d <= remaining means d >= 1/remaining
        # 1/remaining = remaining.denominator / remaining.numerator
        min_d = max(last_denom + 1, -(-remaining.denominator // remaining.numerator))  # ceil
        
        # Upper bound: if we have max_terms terms left, the smallest possible sum
        # is 1/(min_d) + 1/(min_d+1) + ... but we need sum >= remaining
        # So 1/d >= remaining / max_terms roughly, meaning d <= max_terms / remaining
        # More precisely: remaining = sum of 1/d_i where d_i >= min_d
        # The maximum possible sum with max_terms terms starting from min_d is
        # 1/min_d + 1/(min_d+1) + ... but we need exactly remaining
        # Lower bound on d: d >= ceil(1/remaining) already
        # Upper bound on d: if d is too large, 1/d is too small even with max_terms terms
        # remaining <= sum of 1/d_i for i terms, each >= 1/d where d is the largest
        # Actually simpler: 1/d <= remaining, and also remaining >= 1/d (if only one term)
        # For max_terms terms: remaining <= max_terms / min_d (loose upper bound on feasibility)
        # And remaining >= 1/d_max where d_max is the largest denom
        # So d_max >= 1/remaining, which we already have
        # Upper bound: d <= floor(max_terms * (1/remaining)) is wrong
        # Let's just bound: if 1/d < remaining / max_terms, then even max_terms terms of 1/d won't reach remaining
        # So d > max_terms / remaining is infeasible
        # max_terms / remaining = max_terms * remaining.denominator / remaining.numerator
        if remaining.numerator == 0:
            return None
        max_d = (max_terms * remaining.denominator) // remaining.numerator
        if max_d < min_d:
            return None
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            result = try_complete(new_rem, d, max_terms - 1, max(current_max, d))
            if result is not None:
                return [d] + result
        return None
    
    # Strategy: search by number of terms, from 2 up to maybe 6
    # For each, search denominators in increasing order of max
    
    # First, try 2 terms: 1/a + 1/b = 5/31
    # 1/a + 1/b = (a+b)/(ab) = 5/31 => 31(a+b) = 5ab
    # 31a + 31b = 5ab => 31a - 5ab + 31b = 0 => a(31-5b) + 31b = 0
    # a = 31b / (5b - 31)
    # For a to be positive integer: 5b - 31 > 0 => b > 6.2, so b >= 7
    # And a > b (WLOG), so 31b/(5b-31) > b => 31b > 5b^2 - 31b => 62b > 5b^2 => b < 62/5 = 12.4
    # So b in {7, 8, 9, 10, 11, 12}
    for b in range(7, 13):
        denom = 5 * b - 31
        if denom <= 0:
            continue
        num = 31 * b
        if num % denom == 0:
            a = num // denom
            if a > b and a >= 1 and a <= 10000000:
                result = [a, b]
                max_d = max(a, b)
                if best_max is None or max_d < best_max:
                    best_max = max_d
                    best_result = result
    
    # Try 3 terms: search a < b < c, 1/a + 1/b + 1/c = 5/31
    # a must be >= 7 (since 1/a <= 5/31)
    # For each a, 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # This is the same as 2-term problem with target (5a-31)/(31a)
    # b > a, and 1/b < (5a-31)/(31a), so b > 31a/(5a-31)
    # Also 1/b + 1/c = target2, with c > b
    # b(5a-31) + c(5a-31) = 31a... wait let me redo
    # 1/b + 1/c = (5a-31)/(31a)
    # (b+c)/(bc) = (5a-31)/(31a)
    # 31a(b+c) = (5a-31)bc
    # This is complex; let's just iterate b
    
    for a in range(7, 20):  # a can't be too large
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        # Now find b > a such that 1/b < rem2, i.e., b > 1/rem2
        min_b = max(a + 1, -(-rem2.denominator // rem2.numerator))
        # Upper bound for b: rem2 >= 1/b (if c is very large, but c > b so 1/c < 1/b)
        # rem2 = 1/b + 1/c < 2/b, so b < 2/rem2
        max_b = (2 * rem2.denominator) // rem2.numerator
        if max_b < min_b:
            continue
        for b in range(min_b, min(max_b + 1, 1000)):
            if time.time() > deadline:
                break
            rem3 = rem2 - Fraction(1, b)
            if rem3 <= 0:
                break
            # c = 1/rem3
            if rem3.numerator == 1:
                c = rem3.denominator
                if c > b and c <= 10000000:
                    max_d = max(a, b, c)
                    if best_max is None or max_d < best_max:
                        best_max = max_d
                        best_result = [a, b, c]
    
    # Try 4 terms
    for a in range(7, 15):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        min_b = max(a + 1, -(-rem2.denominator // rem2.numerator))
        max_b = (2 * rem2.denominator) // rem2.numerator  # loose bound
        if max_b < min_b:
            continue
        for b in range(min_b, min(max_b + 1, 500)):
            if time.time() > deadline:
                break
            rem3 = rem2 - Fraction(1, b)
            if rem3 <= 0:
                break
            # Now find c > b, d > c with 1/c + 1/d = rem3
            min_c = max(b + 1, -(-rem3.denominator // rem3.numerator))
            max_c = (2 * rem3.denominator) // rem3.numerator
            if max_c < min_c:
                continue
            for c in range(min_c, min(max_c + 1, 200)):
                if time.time() > deadline:
                    break
                rem4 = rem3 - Fraction(1, c)
                if rem4 <= 0:
                    break
                if rem4.numerator == 1:
                    d = rem4.denominator
                    if d > c and d <= 10000000:
                        max_d = max(a, b, c, d)
                        if best_max is None or max_d < best_max:
                            best_max = max_d
                            best_result = [a, b, c, d]
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy from 8
    r = target
    values = []
    while r > 0:
        d = -(-r.denominator // r.numerator)  # ceil(1/r)
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
