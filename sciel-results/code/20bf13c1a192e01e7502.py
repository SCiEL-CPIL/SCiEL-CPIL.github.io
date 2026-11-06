import time
from math import gcd

_MAX_DEN = 10000000
_deadline = None

def _search_k(k):
    """Backtracking search for exactly k distinct denominators >= 5
       whose reciprocal sum is exactly 1."""
    chosen = []
    
    def rec(remaining, last, num, den):
        if time.time() > _deadline:
            return False
        
        if remaining == 0:
            return num == 0
        
        if remaining == 1:
            if num == 0:
                return False
            if den % num == 0:
                d = den // num
                if d > last and d >= 5 and d <= _MAX_DEN:
                    chosen.append(d)
                    return True
            return False
        
        # lower bound on next denominator
        d_low = max(last + 1, 5)
        if num > 0:
            d_low = max(d_low, den // num + 1)
            d_high = min(_MAX_DEN, (remaining * den) // num)
        else:
            d_high = _MAX_DEN
        
        if d_low > d_high:
            return False
        
        # pruning: max possible sum of 'remaining' terms with smallest
        # denominators d_low..d_low+remaining-1 must be >= num/den
        if num > 0:
            # Sum of 1/d for d in [d_low, d_low+remaining-1]
            # Approximate check: remaining / d_low >= num/den
            if remaining * den < num * d_low:
                return False
        
        for d in range(d_low, d_high + 1):
            if time.time() > _deadline:
                return False
            
            # 1/d must be < num/den
            if den >= num * d:
                continue
            
            new_num = num * d - den
            new_den = den * d
            
            if new_num <= 0:
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # pruning: (remaining-1)/(d+1) must be >= new_num/new_den
            if (remaining - 1) * new_den < new_num * (d + 1):
                break
            
            chosen.append(d)
            if rec(remaining - 1, d, new_num, new_den):
                return True
            chosen.pop()
        
        return False

    if rec(k, 4, 1, 1):
        return chosen[:]
    return None

def _construct():
    global _deadline
    _deadline = time.time() + 50.0
    
    # Try to find minimal k
    for k in range(8, 12):
        if time.time() > _deadline:
            break
        sol = _search_k(k)
        if sol is not None:
            return sol
    
    # Fallback: Greedy construction
    result = []
    num, den = 1, 1
    used = set()
    while num > 0 and len(result) < 256:
        if time.time() > _deadline:
            break
        # Find smallest d > last such that 1/d < num/den
        d = den // num + 1
        if d < 5:
            d = 5
        while d in used or d < 5 or d > _MAX_DEN:
            d += 1
        if d > _MAX_DEN:
            break
            
        result.append(d)
        used.add(d)
        num = num * d - den
        den = den * d
        if num < 0:
            break
        g = gcd(num, den)
        num //= g
        den //= g
        
    if result and sum(1/d for d in result) == 1.0:
        return result
    # If greedy failed or didn't sum to 1 (due to overflow or logic), return a known valid one if possible
    # But greedy should work.
    return result

def construct():
    return _construct()
