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
        
        # Check 1/d < num/den
        # This is guaranteed by d > den/num, but integer division might be tricky
        # if num/den is not integer. d_min ensures d >= floor(den/num) + 1.
        # So d > den/num is strictly true?
        # If den/num = 5.1, d_min = 6. 1/6 < 1/5.1? Yes.
        # If den/num = 5.0, d_min = 6. 1/6 < 1/5.0? Yes.
        # So 1/d < num/den is always true for d >= d_min.
        
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
            # d is too small? 
            # As d increases, 1/d decreases, rem increases.
            # Max sum of remaining terms decreases.
            # So if rem is too large, we need to increase d?
            # Wait.
            # rem = N - 1/d.
            # MaxRemSum = (k-1)/(d+1).
            # We need N - 1/d <= (k-1)/(d+1).
            # N <= 1/d + (k-1)/(d+1).
            # As d increases, RHS decreases.
            # So if condition fails (RHS too small), we must DECREASE d?
            # But we are iterating d upwards.
            # If condition fails for current d, it will fail for all larger d.
            # So we can break.
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
    # If search fails, we need a valid solution.
    # The search with tight bounds should work.
    # But let's add a safe fallback if it doesn't.
    # A known solution for sum 1 with denominators >= 5?
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 is > 1.
    # We can use the identity 1/n = 1/(n+1) + 1/(n(n+1)) to split terms.
    # But we need distinct denominators >= 5.
    
    # Let's try a simpler fallback: 
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - excess
    # This is hard to construct manually.
    
    # If _search fails, it might be due to bugs or timeout.
    # Given the tight bounds, it should find a solution for k=8 or k=9 quickly.
    
    raise RuntimeError("No representation found within deadline")
