import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

# Precompute valid L candidates
_VALID_LS = set()
for d in _DIVISORS:
    for k in range(1, _LCM // d + 1):
        _VALID_LS.add(k * d)
_VALID_LS = list(_VALID_LS)

def _find_assignment(L, R, divisors):
    n = len(divisors)
    candidates = []
    for i in range(n):
        d = divisors[i]
        k_start = (L + d - 1) // d
        k_end = (L + R - 1) // d
        if k_start > k_end:
            return None
        cand = list(range(k_start * d, k_end * d + 1, d))
        candidates.append(cand)
        
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assigned = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
        
    if backtrack(0):
        return assigned
    return None

def construct():
    deadline = time.time() + 85
    divisors = _DIVISORS
    
    # Start R from 9 (min range for 10 distinct ints)
    for R in range(9, 1000):
        if time.time() > deadline - 5:
            break
            
        for L in _VALID_LS:
            if time.time() > deadline - 5:
                break
                
            assigned = _find_assignment(L, R, divisors)
            if assigned is not None:
                if max(assigned) <= _MAX_VAL:
                    return [int(v) for v in assigned]
                    
    return [int(d) for d in divisors]
