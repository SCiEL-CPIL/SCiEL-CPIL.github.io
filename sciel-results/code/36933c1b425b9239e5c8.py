"""Combined program: DFS for minimal terms + DP fallback."""
import time
import math

_DEADLINE = 85.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _search(frac_num, frac_den, max_terms, min_den, max_den, out):
    """
    Recursively search for an Egyptian fraction representation.
    frac_num / frac_den is the remaining sum.
    max_terms is the number of terms left to find.
    min_den is the minimum allowed denominator (strictly increasing).
    max_den is the global maximum denominator.
    out is a list to store the denominators.
    Returns True if found, False otherwise.
    """
    if not _deadline_ok():
        return False
    
    # Reduce fraction
    g = _gcd(frac_num, frac_den)
    frac_num //= g
    frac_den //= g

    if max_terms == 0:
        return frac_num == 0

    if max_terms == 1:
        # Need 1/d = frac_num / frac_den
        # d = frac_den / frac_num
        if frac_den % frac_num == 0:
            d = frac_den // frac_num
            if min_den <= d <= max_den:
                out.append(d)
                return True
        return False

    # Determine range for next denominator d
    # 1/d < frac_num / frac_den  => d > frac_den / frac_num
    # d_min = ceil(frac_den / frac_num) + 1
    d_min = max(min_den, frac_den // frac_num + 1)
    
    # Upper bound for d:
    # The remaining sum (frac_num/frac_den - 1/d) must be <= sum of max_terms-1 largest possible fractions
    # Largest possible fractions are 1/(d+1), 1/(d+2), ...
    # So frac_num/frac_den - 1/d <= sum_{j=1}^{max_terms-1} 1/(d+j)
    # Also, we must have d <= max_den.
    
    # A simpler upper bound: d must be such that 1/d is not too small.
    # If 1/d is too small, the remaining sum is too large to be covered by max_terms-1 terms.
    # Max sum of max_terms-1 terms starting from d+1 is approx (max_terms-1)/(d+1).
    # So frac_num/frac_den - 1/d <= (max_terms-1)/(d+1)
    
    for d in range(d_min, max_den + 1):
        if not _deadline_ok():
            return False
        
        # Check if 1/d < frac_num/frac_den
        if frac_num * d >= frac_den:
            continue
            
        # Check if remaining sum is too large
        rem_num = frac_num * d - frac_den
        rem_den = frac_den * d
        
        # Pruning: rem_num/rem_den must be <= sum of 1/(d+1) ... 1/(d+max_terms-1)
        # Approximation for pruning speed:
        # rem_num / rem_den <= (max_terms - 1) / (d + 1)
        # rem_num * (d + 1) <= rem_den * (max_terms - 1)
        if rem_num * (d + 1) > rem_den * (max_terms - 1):
            break # d is too large, even the max possible sum of remaining terms is not enough? 
            # Wait. If d increases, 1/d decreases, so rem increases.
            # If rem is already too large for the smallest possible denominators (d+1...), then increasing d makes rem even larger?
            # No, if d increases, 1/d decreases, so rem = target - 1/d INCREASES.
            # The max sum of remaining terms DECREASES as d increases (denominators d+1... get larger).
            # So if rem > max_possible_sum, then for any larger d, rem will be larger and max_possible_sum will be smaller.
            # So we can break.
            
        out.append(d)
        if _search(rem_num, rem_den, max_terms - 1, d + 1, max_den, out):
            return True
        out.pop()
        
    return False

def _validate(dens):
    if not (1 <= len(dens) <= 256):
        return False
    if len(set(dens)) != len(dens):
        return False
    for d in dens:
        if d < 5 or d > 10000000:
            return False
    # Exact validation using integers
    # Sum 1/d_i = 1
    # LCM of denominators might be huge, but for small k it's manageable.
    # Or just use Fraction for validation only (safe).
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def _dp_fallback():
    """
    Fallback: Find a valid solution using DP on divisors of 2520.
    This guarantees a solution exists.
    """
    target = 2520
    # We want to find a subset of divisors d such that sum(2520/d) = 2520.
    # Minimize the number of terms.
    # States: sum_value -> (count, path)
    # Initialize: sum 0 -> (0, [])
    best_count = {0: 0}
    best_path = {0: []}
    
    divs = [d for d in range(5, target + 1) if target % d == 0]
    
    for d in divs:
        w = target // d
        # Iterate over current states
        for s in list(best_count.keys()):
            new_s = s + w
            if new_s <= target:
                new_count = best_count[s] + 1
                if new_s not in best_count or new_count < best_count[new_s]:
                    best_count[new_s] = new_count
                    best_path[new_s] = best_path[s] + [d]
                    
    if target in best_count:
        return best_path[target]
    return None

def construct():
    # Try to find a minimal solution (likely 8 terms) using DFS
    # We know 7 terms is impossible.
    for k in range(8, 12):
        if not _deadline_ok():
            break
        out = []
        if _search(1, 1, k, 5, 10000000, out):
            if _validate(out):
                return out

    # Fallback to DP
    if _deadline_ok():
        sol = _dp_fallback()
        if sol and _validate(sol):
            return sol
            
    raise RuntimeError("no representation found")
