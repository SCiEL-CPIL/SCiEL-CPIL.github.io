# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    
    def exists_solution(M):
        """Check if there is a solution with max denominator <= M."""
        # Memoization key: (r, d_min)
        # r is Fraction, d_min is int
        memo = {}
        
        def dfs(r, d_min, current):
            if time.time() > deadline:
                return None
            if r == 0:
                return list(current)
            if r < 0:
                return None
            if d_min > M:
                return None
            
            key = (r, d_min)
            if key in memo:
                return memo[key]
            
            # Pruning: max possible sum from d_min to M
            # We can compute this incrementally or just sum
            # For small M, sum is fast
            max_sum = sum(Fraction(1, d) for d in range(d_min, M + 1))
            if max_sum < r:
                memo[key] = None
                return None
            
            # Pruning: if we must pick at least one term, and 1/d_min > r, 
            # we might still pick larger d? No, 1/d decreases.
            # If 1/d_min > r, we can still pick d > d_min.
            # But if 1/M > r, we can't pick any term?
            # No, we need sum = r. If 1/M > r, we can't use M.
            # We need 1/d <= r.
            d_start = max(d_min, (r.denominator + r.numerator - 1) // r.numerator)
            if d_start > M:
                memo[key] = None
                return None
            
            for d in range(d_start, M + 1):
                new_r = r - Fraction(1, d)
                if new_r < 0:
                    break
                current.append(d)
                res = dfs(new_r, d + 1, current)
                current.pop()
                if res is not None:
                    memo[key] = res
                    return res
            
            memo[key] = None
            return None
        
        return dfs(target, 1, [])

    # Binary search for minimal M
    # Lower bound: 7 (since 1/6 > 5/31)
    # Upper bound: 1000 (safe)
    lo, hi = 7, 1000
    best_M = None
    best_sol = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = exists_solution(mid)
        if sol is not None:
            best_M = mid
            best_sol = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_sol is not None:
        # Verify
        s = sum(Fraction(1, d) for d in best_sol)
        if s == target and len(set(best_sol)) == len(best_sol) and max(best_sol) <= 10000000:
            return best_sol
            
    # Fallback to greedy if binary search fails or times out
    r = target
    values = []
    while r > 0 and time.time() < deadline:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
        
    if values:
        return values
    return [1] # Should not happen
# EVOLVE-BLOCK-END
