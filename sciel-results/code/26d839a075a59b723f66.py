import time

_DEADLINE = 85.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _search(num, den, max_terms, min_den, max_den, out):
    """
    Recursively search for an Egyptian fraction representation.
    num/den is the remaining sum (in reduced form).
    max_terms is the number of terms left to find.
    min_den is the minimum allowed denominator (strictly increasing).
    max_den is the global maximum denominator.
    out is a list to store the denominators.
    Returns True if found, False otherwise.
    """
    if not _deadline_ok():
        return False
    
    # Reduce fraction
    g = _gcd(num, den)
    num //= g
    den //= g

    if max_terms == 0:
        return num == 0

    if max_terms == 1:
        # Need 1/d = num/den => d = den/num
        if den % num == 0:
            d = den // num
            if min_den <= d <= max_den:
                out.append(d)
                return True
        return False

    # Determine range for next denominator d
    # 1/d < num/den => d > den/num
    d_min = max(min_den, den // num + 1)
    
    # Upper bound: d must be small enough that 1/d is not too small
    # The remaining sum (num/den - 1/d) must be <= sum of max_terms-1 largest fractions from d+1
    # Approx: num/den - 1/d <= (max_terms-1)/(d+1)
    # Also need 1/d < num/den which is guaranteed by d > den/num
    
    # Simple upper bound: d <= den (since 1/den < num/den when num >= 1)
    # Better: d such that num/den - 1/d > 0 and achievable
    d_max = min(max_den, den)  # conservative
    
    for d in range(d_min, d_max + 1):
        if not _deadline_ok():
            return False
        
        # Check 1/d < num/den
        if num * d >= den:
            continue
            
        # Remaining: num/den - 1/d = (num*d - den) / (den*d)
        rem_num = num * d - den
        rem_den = den * d
        if rem_num <= 0:
            continue
        
        # Check if remaining is achievable with max_terms-1 terms
        # Max possible sum of max_terms-1 terms starting from d+1:
        # 1/(d+1) + 1/(d+2) + ... + 1/(d+max_terms-1)
        # Approximate: (max_terms-1)/(d+1)
        # If rem_num/rem_den > this, then d is too small (but we already ensured 1/d < num/den)
        # Actually we need rem_num/rem_den <= max possible, and also >= min possible
        # Min possible: (max_terms-1)/(d + max_terms - 1) approximately
        # But for pruning, the key check is: can rem be expressed?
        
        # Pruning: if rem_num/rem_den > (max_terms-1)/(d+1), impossible
        # i.e., rem_num * (d+1) > rem_den * (max_terms-1)
        if rem_num * (d + 1) > rem_den * (max_terms - 1):
            # d is too small? No wait, if rem is too large, we need smaller d
            # But we're increasing d, so this should eventually be satisfied
            # Actually this condition means d is too small, but we started from d_min
            # Let me reconsider: as d increases, 1/d decreases, so rem increases
            # So if rem is too large for the remaining terms, we should try larger d
            # But that makes rem even larger... so this is a contradiction
            # Actually: as d increases, 1/d decreases, rem = num/den - 1/d increases
            # The max sum of remaining terms decreases as d increases
            # So there's a window where it's possible
            pass
        
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
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def construct():
    # Try 8 terms first (minimum possible)
    for k in range(8, 13):
        if not _deadline_ok():
            break
        out = []
        if _search(1, 1, k, 5, 10000000, out):
            if len(out) == k and _validate(out):
                return out
    
    # Fallback: hardcoded valid solution
    # Let's use a known construction
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - delta
    # We need to adjust. Let me use a different approach.
    # Known: 1 = 1/2 + 1/3 + 1/7 + 1/42 but denominators must be >= 5
    # Let me construct one:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + correction
    # Actually, let me just return a valid larger set if search fails
    # But 8 terms should be findable...
    
    # Emergency fallback: use DP over a common denominator
    target = 2520
    best = None
    reached = {0: []}
    for d in range(5, target + 1):
        if not _deadline_ok():
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            t2 = total + w
            if t2 <= target and t2 not in reached:
                reached[t2] = values + [d]
                if t2 == target and (best is None or len(values + [d]) < len(best)):
                    best = values + [d]
    if best is not None and _validate(best):
        return best
    
    raise RuntimeError("No representation found within deadline")
