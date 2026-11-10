import math
import bisect
from functools import lru_cache

def construct():
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12
    divisors = [d for d in divisors if d > 12]
    divisors.sort()
    
    if not divisors:
        return [N]

    # We need to find the smallest M such that a subset of distinct divisors <= M multiplies to N.
    # Binary search on M.
    
    def can_form(target, candidates):
        # candidates is a sorted list of divisors <= M
        n = len(candidates)
        if n == 0:
            return None
        
        # Precompute suffix products for pruning
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
        
        @lru_cache(maxsize=None)
        def dfs(i, prod):
            if prod == target:
                return ()
            
            if i >= n or prod > target:
                return None
            
            # Pruning: even if we multiply by all remaining candidates, can we reach target?
            if prod * suffix_prod[i] < target:
                return None
            
            for j in range(i, n):
                d = candidates[j]
                if prod * d > target:
                    break
                
                res = dfs(j + 1, prod * d)
                if res is not None:
                    return (d,) + res
            
            return None
        
        sol = dfs(0, 1)
        return sol

    lo = divisors[0]
    hi = N
    best_sol = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        
        # Get candidates <= mid efficiently
        idx = bisect.bisect_right(divisors, mid)
        candidates = divisors[:idx]
        
        # Quick check: product of all candidates must be >= N
        prod_cand = 1
        for d in candidates:
            prod_cand *= d
            if prod_cand > N:
                break
        if prod_cand < N:
            lo = mid + 1
            continue
            
        sol = can_form(N, candidates)
        if sol is not None:
            best_sol = list(sol)
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_sol is None:
        return [N]
        
    return best_sol
