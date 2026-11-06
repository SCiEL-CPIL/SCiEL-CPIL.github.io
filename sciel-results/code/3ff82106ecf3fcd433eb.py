import time
from math import gcd as _gcd

_MAX_DEN = 10000000


def _construct():
    deadline = time.time() + 80.0
    
    for k in range(4, 257):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol
    
    return _fallback()


def _search_k(k, deadline):
    chosen = []
    
    def rec(remaining, last, num, den):
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
        
        if remaining == 2:
            return _solve_2(last, num, den, chosen, deadline)
        
        d_low = max(last + 1, 5)
        if num > 0:
            d_low = max(d_low, den // num + 1)
        
        if num > 0:
            d_high = min(_MAX_DEN, (remaining * den) // num)
        else:
            d_high = _MAX_DEN
        
        if d_low > d_high:
            return False
        
        for d in range(d_low, d_high + 1):
            if time.time() > deadline:
                return False
            
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


def _solve_2(last, num, den, chosen, deadline):
    if num == 0:
        return False
    
    a_low = max(last + 1, 5)
    a_high = min(_MAX_DEN, (2 * den) // num)
    
    if a_low > a_high:
        return False
    
    for a in range(a_low, a_high + 1):
        if time.time() > deadline:
            return False
        na = num * a - den
        if na <= 0:
            continue
        da = den * a
        if da % na == 0:
            b = da // na
            if b > a and b >= 5 and b <= _MAX_DEN:
                chosen.append(a)
                chosen.append(b)
                return True
    
    return False


def _fallback():
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
