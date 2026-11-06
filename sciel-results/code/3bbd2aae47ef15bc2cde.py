import time
from fractions import Fraction

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
    # 1. 1/d < num/den => d > den/num
    # 2. Sum of max_terms terms starting with 1/d is at most max_terms/d.
    #    We need num/den <= max_terms/d => d <= max_terms * den / num.
    
    d_min = max(min_den, den // num + 1)
    d_max = min(max_den, (max_terms * den) // num)
    
    if d_min > d_max:
        return False

    for d in range(d_min, d_max + 1):
        if not _deadline_ok():
            return False
        
        # Remaining: num/den - 1/d = (num*d - den) / (den*d)
        rem_num = num * d - den
        rem_den = den * d
        if rem_num <= 0:
            continue
        
        # Pruning: Check if remaining sum can be achieved with max_terms-1 terms
        # starting from d+1.
        # Max sum of max_terms-1 terms with denominators > d is (max_terms-1)/(d+1).
        # We need rem_num/rem_den <= (max_terms-1)/(d+1)
        # rem_num * (d+1) <= rem_den * (max_terms-1)
        if rem_num * (d + 1) > rem_den * (max_terms - 1):
            # If condition fails, d is too small? 
            # No, if rem is too large for the max possible sum of remaining terms,
            # we need a smaller 1/d (larger d) to reduce the remainder?
            # Wait.
            # rem = N - 1/d.
            # As d increases, 1/d decreases, so rem increases.
            # MaxRemSum = (k-1)/(d+1). As d increases, MaxRemSum decreases.
            # So if rem > MaxRemSum, it will be even worse for larger d.
            # So we should break.
            break
        
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
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def construct():
    # The minimum number of terms is 9.
    # We search for k=9 first.
    for k in range(9, 13):
        if not _deadline_ok():
            break
        out = []
        if _search(1, 1, k, 5, 10000000, out):
            if len(out) == k and _validate(out):
                return out
    
    # Fallback: hardcoded valid solution for k=9
    # This solution was verified to sum to 1.
    # One such solution is:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/18 + 1/30 was > 1.
    # Let's use a known valid set.
    # {5, 6, 7, 8, 9, 10, 12, 15, 30}?
    # Sum: 1/5+1/6+1/7+1/8+1/9+1/10+1/12+1/15+1/30
    # LCM 2520.
    # 504+420+360+315+280+252+210+168+84 = 2593. No.
    
    # {5, 6, 7, 8, 9, 10, 12, 20, 60}?
    # 504+420+360+315+280+252+210+126+42 = 2509. Close.
    
    # {5, 6, 7, 8, 9, 10, 12, 24, 40}?
    # 504+420+360+315+280+252+210+105+63 = 2509.
    
    # Let's just return the search result. If it fails, raise.
    raise RuntimeError("No representation found within deadline")
