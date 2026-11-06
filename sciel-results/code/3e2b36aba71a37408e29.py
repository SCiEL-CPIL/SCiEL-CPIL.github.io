"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd as _gcd

_MAX_DEN = 10000000


def _construct():
    """
    Find the minimum number of distinct denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    
    Uses integer-arithmetic backtracking with strong pruning.
    """
    deadline = time.time() + 75.0  # safety margin before 90s kill
    
    # Search k from 3 upward
    for k in range(3, 257):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol
    
    # Fallback: guaranteed valid representation
    return _fallback()


def _search_k(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is 1, using integer arithmetic.
    """
    chosen = []
    
    def rec(remaining, last, num, den):
        """
        Find `remaining` distinct denominators > last such that
        their reciprocal sum equals num/den.
        """
        if time.time() > deadline:
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
        
        # remaining >= 2
        d_low = max(last + 1, 5)
        if num > 0:
            d_low = max(d_low, den // num + 1)
        
        if num > 0:
            d_high = min(_MAX_DEN, (remaining * den) // num)
        else:
            d_high = _MAX_DEN
        
        if d_low > d_high:
            return False
        
        # Additional pruning: minimum possible sum of remaining terms
        # with denominators starting from d_low is bounded below by
        # 1/(d_low + remaining - 1) * remaining (very rough)
        # But more useful: max possible sum is sum of 1/(last+1) + ... 
        # Actually, let's use a tighter bound.
        
        for d in range(d_low, d_high + 1):
            if time.time() > deadline:
                return False
            
            # 1/d must be < num/den
            if not (den < num * d):
                continue
            
            new_num = num * d - den
            new_den = den * d
            
            if new_num == 0:
                continue
            
            g = _gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Pruning: max sum of next (remaining-1) terms
            # Each next term has denominator > d, so each <= 1/(d+1)
            # Sum <= (remaining-1)/(d+1)
            # Need (remaining-1)/(d+1) >= new_num/new_den
            if new_num > 0:
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


def _fallback():
    """
    Guaranteed-valid baseline.
    """
    # Use known decomposition: 1 = 1/5 + 1/6 + 1/7 + 1/42 + ... 
    # Let's construct one: 1 = 1/5 + 1/6 + 1/10 + 1/15 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420
    # Actually, let me use a simpler approach: 
    # 1 = 1/2 + 1/3 + 1/6, but 2 < 5.
    # 1 = 1/5 + 4/5. Need 4/5 as sum of distinct unit fractions with denom >= 6.
    # 4/5 = 1/2 + 3/10, but 2 < 5.
    # Let's use: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520 + 1/5040
    # Check: 1/5+1/6+1/7 = (42+35+30)/210 = 107/210. 1 - 107/210 = 103/210.
    # This is getting complicated. Let me just use a DP approach with a smaller target.
    
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")


def construct():
    return _construct()
# EVOLVE-BLOCK-END
