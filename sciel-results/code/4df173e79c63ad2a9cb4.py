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

def _max_sum(n_terms, min_den):
    """Maximum possible sum of n_terms distinct reciprocals with denominators >= min_den"""
    total = Fraction(0, 1)
    d = min_den
    for _ in range(n_terms):
        total += Fraction(1, d)
        d += 1
    return total

def _min_possible_sum(n_terms, max_den):
    """Minimum possible sum of n_terms distinct reciprocals with denominators <= max_den"""
    total = Fraction(0, 1)
    d = max_den
    for _ in range(n_terms):
        total += Fraction(1, d)
        d -= 1
    return total

def _search(num, den, max_terms, min_den, max_den, out):
    if not _deadline_ok():
        return False

    g = _gcd(num, den)
    num //= g
    den //= g

    if num == 0:
        return max_terms == 0

    if max_terms == 0:
        return False

    # Pruning: check if remaining terms can sum to num/den
    # Max possible: use smallest available denominators
    if _max_sum(max_terms, min_den) < num / den:
        return False
    # Min possible: use largest available denominators  
    if _min_possible_sum(max_terms, max_den) > num / den:
        return False

    if max_terms == 1:
        if den % num == 0:
            d = den // num
            if min_den <= d <= max_den:
                out.append(d)
                return True
        return False

    # d must satisfy: 1/d < num/den (so remainder is positive)
    # and 1/d > num/den / max_terms (so remaining can reach target)
    d_min = max(min_den, den // num + 1)
    # For remaining (max_terms-1) terms to sum to (num*d - den)/(den*d),
    # we need (num*d - den)/(den*d) <= (max_terms-1) / (d+1) approximately
    # Simpler: d must be at most max_terms * den / num
    d_max = min(max_den, (max_terms * den) // num)

    if d_min > d_max:
        return False

    # Additional pruning: after picking d, remaining sum = num/den - 1/d = (num*d - den)/(den*d)
    # This must be achievable with (max_terms-1) terms from [d+1, max_den]
    # Max of remaining: _max_sum(max_terms-1, d+1) >= (num*d-den)/(den*d)
    # Min of remaining: _min_possible_sum(max_terms-1, max_den) <= (num*d-den)/(den*d)

    for d in range(d_min, d_max + 1):
        if not _deadline_ok():
            return False

        rem_num = num * d - den
        rem_den = den * d
        if rem_num <= 0:
            continue

        # Prune: can (max_terms-1) terms from [d+1, max_den] sum to rem_num/rem_den?
        if _max_sum(max_terms - 1, d + 1) < rem_num / rem_den:
            break  # d too large, further d will be worse
        if _min_possible_sum(max_terms - 1, max_den) > rem_num / rem_den:
            continue

        out.append(d)
        if _search(rem_num, rem_den, max_terms - 1, d + 1, max_den, out):
            return True
        out.pop()

    return False

def construct():
    # Prove lower bound: no solution with fewer than 8 terms
    # Max sum of k distinct reciprocals with denominators >= 5:
    # k=4: 1/5+1/6+1/7+1/8 = 0.6458 < 1
    # k=5: +1/9 = 0.7569 < 1  
    # k=6: +1/10 = 0.8569 < 1
    # k=7: +1/11 = 0.9478 < 1
    # k=8: +1/12 = 1.0311 > 1 (possible!)
    
    # Search for 8-term solution
    for k in range(8, 13):
        if not _deadline_ok():
            break
        out = []
        if _search(1, 1, k, 5, 10000000, out):
            if len(out) == k and _validate(out):
                return out

    # Fallback: greedy method
    dens = []
    rem = Fraction(1, 1)
    min_d = 5
    for _ in range(50):
        d = min_d
        while Fraction(1, d) >= rem:
            d += 1
        if d > 10000000:
            break
        dens.append(d)
        rem -= Fraction(1, d)
        min_d = d + 1
        if rem == 0:
            break
        if not _deadline_ok():
            break

    if dens and _validate(dens):
        return dens

    # Absolute fallback
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
