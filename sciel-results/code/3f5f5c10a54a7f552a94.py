"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

_MAX_DEN = 10000000


def _max_fill_exact(last, count):
    """Exact max reciprocal sum with `count` distinct denominators > last, as Fraction."""
    s = Fraction(0, 1)
    for j in range(1, count + 1):
        s += Fraction(1, last + j)
    return s


def _search_k(k, deadline):
    """Backtracking for exactly k distinct denominators >= 5 summing to 1."""
    
    def rec(remaining, last, num, den):
        if time.time() > deadline:
            return None
        
        if num > den:
            return None
            
        if remaining == 0:
            return None if num != den else []
            
        # Prune: can't reach 1 even with smallest available denominators
        max_possible = _max_fill_exact(last, remaining)
        if num + den * max_possible < den:
            return None
            
        if remaining == 1:
            diff = den - num
            if diff <= 0:
                return None
            if den % diff == 0:
                d = den // diff
                if d > last and d >= 5 and d <= _MAX_DEN:
                    return [d]
            return None

        diff = den - num
        if diff <= 0:
            return None
            
        d_low = max(last + 1, (den + diff - 1) // diff)
        
        d = d_low
        guard = 0
        while d <= _MAX_DEN:
            guard += 1
            if guard > 500000 or time.time() > deadline:
                return None
                
            new_num = num * d + den
            new_den = den * d
            
            g = math.gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
                
            if new_num > new_den:
                d += 1
                continue
                
            # Prune: check if we can still reach target
            max_possible_next = _max_fill_exact(d, remaining - 1)
            if new_num + new_den * max_possible_next < new_den:
                break
                
            res = rec(remaining - 1, d, new_num, new_den)
            if res is not None:
                return [d] + res
                
            d += 1
            
        return None

    return rec(k, 4, 0, 1)


def _fallback():
    """Guaranteed-valid baseline solution."""
    # Known identity: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - 1/1820 + ...
    # Let's use a verified solution with 9 terms
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/1820?
    # Let me compute: sum of 1/5..1/12 = 11077/10810 ≈ 1.0247
    # Actually, let me just use a subset-sum approach on a common denominator
    
    # Use lcm(5..20) = 232792560, but that's too large for DP
    # Instead, use a greedy + adjustment approach
    
    # Known valid: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - 1/1820
    # But we need positive terms only. Let me find a valid 9-term solution.
    
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 11077/10810
    # We need to replace some terms to get exactly 1.
    
    # Alternative: use the fact that 1 = 1/2 + 1/3 + 1/6, but denominators must be >= 5.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - 1/1820 doesn't work directly.
    
    # Let me just do a simple DP on divisors of a moderate common denominator.
    target = 2520  # lcm(1..10)
    # Divisors of 2520 that are >= 5:
    divisors = []
    for d in range(5, target + 1):
        if target % d == 0:
            divisors.append(d)
    
    # DP: can we reach sum=target using weights target//d?
    # Use bitset or set-based DP
    reachable = {0: []}
    for d in divisors:
        w = target // d
        for total in list(reachable.keys()):
            new_total = total + w
            if new_total <= target and new_total not in reachable:
                reachable[new_total] = reachable[total] + [d]
        if target in reachable:
            return reachable[target]
    
    # Try larger common denominator
    for target in [5040, 10080, 25200, 50400]:
        divisors = []
        for d in range(5, min(target + 1, 100000)):
            if target % d == 0:
                divisors.append(d)
        
        reachable = {0: []}
        for d in divisors:
            w = target // d
            for total in list(reachable.keys()):
                new_total = total + w
                if new_total <= target and new_total not in reachable:
                    reachable[new_total] = reachable[total] + [d]
            if target in reachable:
                return reachable[target]
    
    raise RuntimeError("no solution found")


def construct():
    deadline = time.time() + 55.0
    
    # k < 8 is impossible
    for k in range(8, 15):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol
    
    return _fallback()
# EVOLVE-BLOCK-END
