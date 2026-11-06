"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_MAX_DEN = 10000000


def _max_fill_float(last, count):
    """Approximate max reciprocal sum with `count` distinct denominators > last."""
    s = 0.0
    for j in range(1, count + 1):
        s += 1.0 / (last + j)
    return s


def _search(k, deadline):
    """Backtracking for exactly k distinct denominators >= 5 summing to 1."""
    
    def rec(remaining, last, num, den):
        if time.time() > deadline:
            return None
        
        if num > den:
            return None
            
        if remaining == 0:
            return None if num != den else []
            
        # Prune: can't reach 1 even with smallest available denominators
        max_possible = _max_fill_float(last, remaining)
        if (num / den) + max_possible < 1.0:
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
            if guard > 200000 or time.time() > deadline:
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
                
            if (new_num / new_den) + _max_fill_float(d, remaining - 1) < 1.0:
                break
                
            res = rec(remaining - 1, d, new_num, new_den)
            if res is not None:
                return [d] + res
                
            d += 1
            
        return None

    return rec(k, 4, 0, 1)


def _fallback():
    """Guaranteed-valid baseline solution using subset DP over divisors of a common denominator."""
    # Use a larger common denominator for more flexibility
    # 2520 = lcm(1..10), but let's use a larger one for more options
    # Actually, let's use the approach from inspiration 1: subset DP over divisors
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and (total + w) not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    # If 2520 doesn't work, try larger common denominators
    for target in [5040, 10080, 25200, 50400, 100800]:
        reached = {0: []}
        for d in range(5, min(target + 1, 100001)):
            if target % d != 0:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and (total + w) not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
    
    raise RuntimeError("no solution found")


def construct():
    deadline = time.time() + 55.0
    
    # k < 8 is impossible (max sum of 7 largest allowed reciprocals < 1)
    # k=8: max sum = 1/5+1/6+...+1/12 ≈ 1.02 > 1, so possible
    for k in range(8, 257):
        if time.time() > deadline:
            break
        sol = _search(k, deadline)
        if sol is not None:
            return sol
    
    return _fallback()
# EVOLVE-BLOCK-END
