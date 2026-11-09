"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 80.0  # Safety margin for 90s limit

def _validate_solution(denoms):
    """Check if denominators are valid and sum to TARGET."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    total = sum(Fraction(1, d) for d in denoms)
    return total == TARGET

def _try_solution(denoms):
    """Validate and return max denominator if valid, else None."""
    if _validate_solution(denoms):
        return max(denoms)
    return None

def _search_3term(max_c, start_time, deadline):
    """
    Exhaustive search for 3-term Egyptian fraction 1/a+1/b+1/c = 5/31
    with a < b < c <= max_c. Returns best solution found (min max denominator)
    or None.
    """
    best = None
    best_max = max_c + 1
    
    # a must satisfy: 1/a < 5/31, so a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 (since b,c > a, so 1/b+1/c < 2/a, thus 1/a + 2/a > 5/31)
    # 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    for a in range(7, 19):
        if time.time() - start_time > deadline:
            break
        r_ab = TARGET - Fraction(1, a)
        if r_ab <= 0:
            continue
        # Now 1/b + 1/c = r_ab with a < b < c <= max_c
        # 1/b < r_ab => b > 1/r_ab
        # Also 2/b > r_ab => b < 2/r_ab
        b_min = max(a + 1, (r_ab.denominator + r_ab.numerator - 1) // r_ab.numerator + 1)
        b_max = min(2 * r_ab.denominator // r_ab.numerator, max_c - 1)
        
        if b_min > b_max:
            continue
        
        for b in range(b_min, b_max + 1):
            if time.time() - start_time > deadline:
                break
            r_c = r_ab - Fraction(1, b)
            if r_c <= 0:
                break
            if r_c.numerator == 1:
                c = r_c.denominator
                if c > b and c <= max_c:
                    md = c
                    if md < best_max:
                        best_max = md
                        best = [a, b, c]
    
    return best, best_max

def _search_4term(max_d, start_time, deadline):
    """
    Search for 4-term Egyptian fraction with all denominators <= max_d.
    Uses systematic enumeration with strong pruning.
    """
    best = None
    best_max = max_d + 1
    
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d <= max_d
    # a >= 7 (since 4/a > 5/31 => a < 124/5 = 24.8)
    # a > 31/5 = 6.2 => a >= 7
    # 4/a > 5/31 => a < 124/5 => a <= 24
    
    for a in range(7, 25):
        if time.time() - start_time > deadline:
            break
        r_bcd = TARGET - Fraction(1, a)
        if r_bcd <= 0:
            continue
        # b > a, and 3/b > r_bcd => b < 3/r_bcd
        # 1/b < r_bcd => b > 1/r_bcd
        b_min = max(a + 1, (r_bcd.denominator + r_bcd.numerator - 1) // r_bcd.numerator + 1)
        b_max = min(3 * r_bcd.denominator // r_bcd.numerator, max_d - 2)
        
        if b_min > b_max:
            continue
        
        for b in range(b_min, b_max + 1):
            if time.time() - start_time > deadline:
                break
            r_cd = r_bcd - Fraction(1, b)
            if r_cd <= 0:
                break
            # c > b, 2/c > r_cd => c < 2/r_cd
            # 1/c < r_cd => c > 1/r_cd
            c_min = max(b + 1, (r_cd.denominator + r_cd.numerator - 1) // r_cd.numerator + 1)
            c_max = min(2 * r_cd.denominator // r_cd.numerator, max_d - 1)
            
            if c_min > c_max:
                continue
            
            for c in range(c_min, c_max + 1):
                if time.time() - start_time > deadline:
                    break
                r_d = r_cd - Fraction(1, c)
                if r_d <= 0:
                    break
                if r_d.numerator == 1:
                    d = r_d.denominator
                    if d > c and d <= max_d:
                        md = d
                        if md < best_max:
                            best_max = md
                            best = [a, b, c, d]
    
    return best, best_max

def _solve():
    """
    Search for a representation of 5/31 with minimal largest denominator.
    Strategy:
    1. Start with known solution [8, 31, 248].
    2. Systematically search for 3-term solutions with smaller max.
    3. Systematically search for 4-term solutions with smaller max.
    """
    start_time = time.time()
    
    # Known solution
    best_solution = [8, 31, 248]
    best_max_d = 248
    
    # Phase 1: Search for 3-term solutions with c < best_max_d
    # Try decreasing max_c from best_max_d - 1 downward
    for max_c in range(best_max_d - 1, 200, -1):
        if time.time() - start_time > DEADLINE:
            break
        sol, found_max = _search_3term(max_c, start_time, DEADLINE)
        if sol is not None:
            best_solution = sol
            best_max_d = found_max
            break  # Found a 3-term solution, this is likely optimal for 3 terms
    
    # If no 3-term found below 248, search 4-term
    if best_max_d >= 248:
        for max_d in range(min(best_max_d - 1, 200), 150, -1):
            if time.time() - start_time > DEADLINE:
                break
            sol, found_max = _search_4term(max_d, start_time, DEADLINE)
            if sol is not None:
                best_solution = sol
                best_max_d = found_max
                break
    
    # If still 248, try broader 4-term search
    if best_max_d >= 248:
        sol, found_max = _search_4term(247, start_time, DEADLINE)
        if sol is not None:
            best_solution = sol
            best_max_d = found_max
    
    return best_solution

def construct():
    """
    Entry point. Returns a list of distinct denominators summing to 5/31.
    Minimizes the largest denominator.
    """
    return _solve()
# EVOLVE-BLOCK-END
