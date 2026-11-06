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

    if max_terms == 2:
        # Need 1/d1 + 1/d2 = frac_num / frac_den, with d1 > min_den, d2 > d1
        # 1/d2 = frac_num/frac_den - 1/d1 = (frac_num*d1 - frac_den) / (frac_den*d1)
        # So d2 = frac_den*d1 / (frac_num*d1 - frac_den)
        # Let N = frac_num, D = frac_den
        # d2 = D*d1 / (N*d1 - D)
        # Let rem = N*d1 - D. We need rem > 0 and rem divides D*d1.
        # Also d2 > d1.
        
        # Range for d1:
        # 1/d1 < frac_num/frac_den => d1 > frac_den/frac_num
        # d1_min = max(min_den, frac_den // frac_num + 1)
        
        # Also 1/d1 + 1/(d1+1) >= frac_num/frac_den (smallest possible sum for 2 terms starting at d1)
        # Actually, we just iterate d1 and check if d2 is integer and > d1.
        
        d1_min = max(min_den, frac_den // frac_num + 1)
        
        # Upper bound for d1:
        # 1/d1 + 1/(d1+1) >= frac_num/frac_den
        # 2d1+1 / (d1(d1+1)) >= N/D
        # This is a quadratic inequality. For large d1, LHS ~ 2/d1.
        # So 2/d1 >= N/D => d1 <= 2D/N.
        # Let's just use a reasonable upper bound or break condition.
        
        for d1 in range(d1_min, max_den + 1):
            if not _deadline_ok():
                return False
            
            rem_num = frac_num * d1 - frac_den
            if rem_num <= 0:
                continue
            
            rem_den = frac_den * d1
            
            if rem_den % rem_num != 0:
                continue
            
            d2 = rem_den // rem_num
            
            if d2 <= d1:
                continue
            
            if d2 > max_den:
                break # d2 will only get larger as d1 increases? 
                # Let's check: d2(d1) = D*d1 / (N*d1 - D).
                # Derivative wrt d1: D*(N*d1-D - D) / (N*d1-D)^2 = D*(N*d1-2D)/(...)
                # If d1 > 2D/N, derivative is positive, so d2 increases.
                # Since d1_min > D/N, we are likely in the region where d2 increases or is close.
                # If d2 > max_den, it's safe to break if we know d2 is increasing.
                # For d1 > 2D/N, d2 is increasing.
                # If d1 <= 2D/N, d2 might decrease. But d1_min is roughly D/N.
                # Let's be safe and not break immediately, but check next few?
                # Actually, if d2 > max_den, and d2 is increasing, we can break.
                # If d2 is decreasing, we might find a valid d2 later.
                # However, d1 is increasing.
                # Let's just continue for now, or break if d1 is large enough.
                # Given the constraints, d1 won't be huge.
                pass
            
            out.append(d1)
            out.append(d2)
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
