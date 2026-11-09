"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

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

def _search_bounded(max_d_limit, start_time, deadline):
    """
    Search for Egyptian fraction of TARGET using only denominators <= max_d_limit.
    Returns a valid solution or None.
    """
    # Recursive search with pruning
    def search(remaining, used, min_d, depth, max_depth):
        if time.time() - start_time > deadline:
            return None
        if remaining == 0:
            return []
        if depth >= max_depth:
            return None
        if remaining < 0:
            return None
        
        # Lower bound on next denominator: ceil(1/remaining)
        # Upper bound: max_d_limit
        # Also must be > min_d and not in used
        
        # If remaining is a unit fraction, check directly
        if remaining.numerator == 1:
            d = remaining.denominator
            if d > min_d and d <= max_d_limit and d not in used:
                return [d]
            return None
        
        # Determine range for next denominator
        # 1/d < remaining => d > 1/remaining
        # 1/d > remaining/(max_depth - depth) for feasibility (rough bound)
        
        # Greedy lower bound
        d_min = max(min_d + 1, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        if d_min < min_d + 1:
            d_min = min_d + 1
        
        d_max = max_d_limit
        
        # Additional pruning: if we need k more terms, each <= 1/d_min
        # remaining <= k / d_min
        k = max_depth - depth
        if k > 0 and d_min * k < remaining.numerator * 1:  # rough
            pass  # skip this check, it's not tight enough
        
        for d in range(d_min, d_max + 1):
            if d in used:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                return [d]
            result = search(new_remaining, used | {d}, d, depth + 1, max_depth)
            if result is not None:
                return [d] + result
        return None
    
    # Try with increasing max_depth
    for max_depth in range(2, 10):
        result = search(TARGET, set(), 0, 0, max_depth)
        if result is not None:
            return result
    return None

def _solve():
    """
    Search for a representation of 5/31 with minimal largest denominator.
    Strategy:
    1. Try known good solutions.
    2. Systematic bounded search for 3-term, 4-term, etc. representations.
    3. Track the solution with minimum max(denominators).
    """
    start_time = time.time()
    
    # Known solution: [8, 31, 248] -> max = 248
    known_solutions = [
        [8, 31, 248],
        [7, 55, 11935, 3979, 7958],  # from greedy, may not be optimal
    ]
    
    best_solution = None
    best_max_d = MAX_DENOM + 1
    
    # Check known solutions
    for sol in known_solutions:
        md = _try_solution(sol)
        if md is not None and md < best_max_d:
            best_max_d = md
            best_solution = sol
    
    # Systematic search: try bounded search with decreasing max_d_limit
    # Start from a reasonable bound and decrease
    # We know max_d_limit=248 works, so try to find something better
    
    # Try 3-term solutions first
    # For 3 terms: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a >= 7, a <= 248 (from known solution)
    # b > a, c > b
    
    search_start = time.time()
    
    # Search for 3-term solutions with c <= best_max_d - 1
    target_max = best_max_d - 1 if best_solution else 500
    
    for c in range(200, target_max + 1):
        if time.time() - start_time > DEADLINE:
            break
        for a in range(7, min(c - 1, 100)):
            if time.time() - start_time > DEADLINE:
                break
            r_ab = TARGET - Fraction(1, c)
            if r_ab <= 0:
                continue
            # Now need 1/a + 1/b = r_ab with a < b < c
            # 1/a < r_ab => a > 1/r_ab
            a_min = max(7, (r_ab.denominator + r_ab.numerator - 1) // r_ab.numerator)
            a_max = min(c - 2, 200)  # limit a for efficiency
            for b in range(a + 1, c):
                if time.time() - start_time > DEADLINE:
                    break
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                    sol = [a, b, c]
                    md = max(sol)
                    if md < best_max_d:
                        best_max_d = md
                        best_solution = sol
    
    # Search for 4-term solutions if we didn't find a great 3-term
    if best_max_d > 200:
        # Try bounded search
        result = _search_bounded(min(best_max_d - 1, 500), start_time, DEADLINE)
        if result is not None:
            md = max(result)
            if md < best_max_d:
                best_max_d = md
                best_solution = result
    
    # If still not great, try wider 2-step greedy search
    if best_max_d > 300:
        for d1 in range(7, 30):
            if time.time() - start_time > DEADLINE:
                break
            r1 = TARGET - Fraction(1, d1)
            if r1 <= 0:
                continue
            used = {d1}
            for d2 in range(d1 + 1, min(d1 + 100, best_max_d)):
                if time.time() - start_time > DEADLINE:
                    break
                if d2 in used:
                    continue
                r2 = r1 - Fraction(1, d2)
                if r2 <= 0:
                    continue
                used2 = used | {d2}
                # Greedy complete
                completion = []
                r = r2
                last_d = d2
                ok = True
                while r > 0:
                    if r.numerator == 0:
                        break
                    d = (r.denominator + r.numerator - 1) // r.numerator
                    while d in used2 or d <= last_d:
                        d += 1
                    if d > best_max_d - 1:
                        ok = False
                        break
                    if len(completion) + len(used2) >= MAX_TERMS:
                        ok = False
                        break
                    completion.append(d)
                    used2.add(d)
                    r -= Fraction(1, d)
                    last_d = d
                if ok and r == 0:
                    full_solution = [d1, d2] + completion
                    max_d = max(full_solution)
                    if max_d < best_max_d:
                        best_max_d = max_d
                        best_solution = full_solution
    
    if best_solution is None:
        # Fallback: pure greedy
        r = TARGET
        used = set()
        last_d = 0
        vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used or d <= last_d:
                d += 1
            if d > MAX_DENOM:
                break
            vals.append(d)
            used.add(d)
            r -= Fraction(1, d)
            last_d = d
        best_solution = vals
    
    return best_solution

def construct():
    """
    Entry point. Returns a list of distinct denominators summing to 5/31.
    Minimizes the largest denominator.
    """
    return _solve()
# EVOLVE-BLOCK-END
