"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _factorize(n):
    """Return prime factorization as list of (prime, exponent) pairs."""
    factors = []
    d = 2
    while d * d <= n:
        cnt = 0
        while n % d == 0:
            cnt += 1
            n //= d
        if cnt > 0:
            factors.append((d, cnt))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def _all_divisors(n):
    """Generate all divisors of n."""
    fac = _factorize(n)
    divs = [1]
    for p, e in fac:
        new_divs = []
        pw = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    return sorted(divs)

def construct():
    target = factorial(12)  # 479001600
    
    # Precompute all divisors > 12
    all_divs = [d for d in _all_divisors(target) if d > 12]
    all_divs.sort()
    
    best_max = target
    best_factors = [target]
    
    def dfs(remaining, start_idx, chosen, path):
        nonlocal best_max, best_factors
        
        if _time_left() < 0.5:
            return
        
        if remaining == 1:
            if path:
                max_f = max(path)
                if max_f < best_max:
                    best_max = max_f
                    best_factors = list(path)
            return
        
        if len(path) >= 12:
            return
        
        for i in range(start_idx, len(all_divs)):
            d = all_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            if d in chosen:
                continue
            
            new_max = max(d, max(path) if path else 0)
            
            # Pruning: if new_max >= best_max and this isn't the last factor, skip
            rem2 = remaining // d
            if new_max >= best_max and rem2 != 1:
                continue
            
            # Pruning: if rem2 < 13 and rem2 != 1, can't factor further
            if rem2 != 1 and rem2 < 13:
                continue
            
            chosen.add(d)
            path.append(d)
            
            dfs(rem2, i + 1, chosen, path)
            
            path.pop()
            chosen.remove(d)
    
    chosen = set()
    path = []
    dfs(target, 0, chosen, path)
    
    return best_factors
# EVOLVE-BLOCK-END
