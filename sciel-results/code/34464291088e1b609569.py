import math
import time

T12 = math.factorial(12)
PRIMES = [2, 3, 5, 7, 11]
EXP_TOTAL = (10, 5, 2, 1, 1)

def _exp_vector(n):
    v = [0] * 5
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            n //= p
            v[i] += 1
    return tuple(v)

# Precompute all divisors
ALL_DIVS = []
ALL_EXP_VECS = []
def _init_divisors():
    limit = math.isqrt(T12)
    seen = set()
    for i in range(1, limit + 1):
        if T12 % i == 0:
            for d in (i, T12 // i):
                if d > 12 and d not in seen:
                    seen.add(d)
                    ALL_DIVS.append(d)
    ALL_DIVS.sort()
    for d in ALL_DIVS:
        ALL_EXP_VECS.append(_exp_vector(d))

_init_divisors()

def _find_factors(M, deadline):
    """
    Find a set of distinct factors from ALL_DIVS <= M with product T12.
    Returns list of factors or None.
    """
    indices = [i for i, d in enumerate(ALL_DIVS) if d <= M]
    if not indices:
        return None
    
    # Sort ascending - try smaller factors first
    indices.sort(key=lambda i: ALL_DIVS[i])
    
    n = len(indices)
    failed = set()
    solution = []
    
    def dfs(pos, rem_exp):
        if time.time() > deadline:
            return False
        if all(e == 0 for e in rem_exp):
            return True
        if pos >= n:
            return False
        state = (pos, rem_exp)
        if state in failed:
            return False
        i = indices[pos]
        ev = ALL_EXP_VECS[i]
        can_take = True
        new_rem = [0]*5
        for k in range(5):
            if ev[k] > rem_exp[k]:
                can_take = False
                break
            new_rem[k] = rem_exp[k] - ev[k]
        if can_take:
            solution.append(ALL_DIVS[i])
            if dfs(pos + 1, tuple(new_rem)):
                return True
            solution.pop()
        if dfs(pos + 1, rem_exp):
            return True
        failed.add(state)
        return False
    
    if dfs(0, EXP_TOTAL):
        return solution
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    lo = 13
    hi = T12
    best_M = T12
    best_factors = [T12]
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        factors = _find_factors(mid, deadline)
        if factors:
            best_M = mid
            best_factors = factors
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_factors:
        prod = 1
        for f in best_factors:
            prod *= f
        if prod == T12 and len(set(best_factors)) == len(best_factors) and all(f > 12 for f in best_factors) and 1 <= len(best_factors) <= 12:
            return best_factors
            
    return [T12]
