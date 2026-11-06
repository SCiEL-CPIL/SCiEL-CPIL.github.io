import time
from fractions import Fraction

# --- explicit deadline guard -------------------------------------------------
_DEADLINE = 85.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE

def _search(target, terms, min_den, max_den, out):
    """Recursively build an Egyptian fraction for the rational target.
    Returns True and fills out[] with 'terms' denominators (each in
    [min_den, max_den]) whose reciprocal sum equals target, or False."""
    if not _deadline_ok():
        return False
    
    if terms == 1:
        # target must be 1/d
        # d = 1 / target
        d_frac = Fraction(1, 1) / target
        if d_frac.denominator == 1:
            d_int = d_frac.numerator
            if min_den <= d_int <= max_den:
                out.append(d_int)
                return True
        return False

    # Determine range for next denominator d
    # We need 1/d < target, so d > 1/target
    x = Fraction(1, 1) / target
    # d must be strictly greater than x
    start = max(min_den, x.numerator // x.denominator + 1)
    
    # Upper bound for d is max_den, but we break early if impossible
    # Condition for break:
    # target - 1/d > sum_{i=1}^{terms-1} 1/(d+i)
    # rem_target > max_possible_sum
    
    for d in range(start, max_den + 1):
        if not _deadline_ok():
            return False
        
        # 1/d must be less than target
        # This is guaranteed by d > 1/target, but let's be safe
        if Fraction(1, d) >= target:
            continue

        rem_target = target - Fraction(1, d)
        
        # Calculate max possible sum of remaining 'terms-1' fractions
        # starting from denominator d+1
        max_sum = Fraction(0, 1)
        for i in range(1, terms):
            max_sum += Fraction(1, d + i)
            
        if rem_target > max_sum:
            break # d is too large, remaining sum cannot be achieved
            
        out.append(d)
        if _search(rem_target, terms - 1, d + 1, max_den, out):
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
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def construct():
    # The minimum number of terms is 8, as sum(1/5..1/11) < 1 and sum(1/5..1/12) > 1
    # We search for k=8 first, then 9, etc.
    for k in range(8, 16):
        if not _deadline_ok():
            break
        out = []
        if _search(Fraction(1, 1), k, 5, 10000000, out):
            if len(out) == k and _validate(out):
                return out
    
    # Fallback: If search fails (should not happen for k=8), return a known valid larger set
    # A known 10-term solution:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + ... 
    # Actually, just return the result of a simple greedy if we must, but search should work.
    # To be absolutely safe against timeout in pathological cases (though unlikely),
    # we can hardcode a valid 8-term solution if we find one during development.
    # Since I can't run it here, I rely on the search.
    
    # If all else fails, raise an error
    raise RuntimeError("No representation found within deadline")
