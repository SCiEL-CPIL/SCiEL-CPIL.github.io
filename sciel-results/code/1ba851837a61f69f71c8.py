"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from fractions import Fraction
import time

def _dfs(rem, min_d, max_d, remaining_terms, path, best, start_time, time_limit):
    """
    Try to represent rem (a Fraction) as a sum of remaining_terms unit fractions,
    with each denominator in [min_d, max_d], all distinct, and in increasing order.
    """
    if time.time() - start_time > time_limit:
        return False
    
    if remaining_terms == 0:
        if rem == 0:
            if not best[0] or len(path) < len(best[0]):
                best[0] = list(path)
            return True
        return False
    
    if rem <= 0:
        return False
    
    p = rem.numerator
    q = rem.denominator
    
    # Lower bound for d: d >= ceil(q / p)
    lo = (q + p - 1) // p
    if lo < min_d:
        lo = min_d
    
    # Upper bound
    hi = max_d
    
    # Cap hi to avoid excessive iteration
    upper_cap = (q // p) * remaining_terms + 10
    if upper_cap < hi:
        hi = upper_cap
    if hi > max_d:
        hi = max_d
    
    if lo > hi:
        return False
    
    for d in range(lo, hi + 1):
        if d > max_d:
            break
        # Check 1/d <= rem => d >= q/p, already ensured by lo
        # Check new_rem > 0
        if d * p <= q:  # 1/d >= p/q, so rem - 1/d <= 0
            if d * p == q:
                if remaining_terms == 1:
                    path.append(d)
                    if not best[0] or len(path) < len(best[0]):
                        best[0] = list(path)
                    path.pop()
                    return True
            continue
        
        new_rem = rem - Fraction(1, d)
        
        # Pruning: new_rem must be <= (remaining_terms - 1) / (d + 1)
        if remaining_terms - 1 > 0:
            max_possible = Fraction(remaining_terms - 1, d + 1)
            if new_rem > max_possible:
                continue
        
        path.append(d)
        if _dfs(new_rem, d + 1, max_d, remaining_terms - 1, path, best, start_time, time_limit):
            path.pop()
            return True
        path.pop()
    
    return False

def construct():
    target = Fraction(1, 1)
    max_denom = 10000000
    best = [None]
    start_time = time.time()
    time_limit = 50  # seconds, leave 10s for overhead
    
    # We know at least 8 terms are needed
    for num_terms in range(8, 16):
        if time.time() - start_time > time_limit:
            break
        path = []
        if _dfs(target, 5, max_denom, num_terms, path, best, start_time, time_limit):
            result = best[0]
            if result and len(result) >= 1 and len(result) <= 256:
                total = sum(Fraction(1, d) for d in result)
                if total == 1 and all(5 <= d <= max_denom for d in result) and len(set(result)) == len(result):
                    return result
            break
    
    # Fallback: greedy Egyptian fraction
    def greedy_egyptian(n, d, min_denom=5):
        terms = []
        while d != 0:
            denom = max(min_denom, (d + n - 1) // n)
            if denom > 10000000:
                break
            terms.append(denom)
            new_n = n * denom - d
            new_d = d * denom
            g = gcd(new_n, new_d)
            n = new_n // g
            d = new_d // g
            min_denom = denom + 1
        return terms
    
    fallback = greedy_egyptian(1, 1, 5)
    if fallback and all(5 <= d <= 10000000 for d in fallback) and len(fallback) <= 256:
        total = sum(Fraction(1, d) for d in fallback)
        if total == 1:
            return fallback
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
