import time
from math import factorial

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    target = factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    divisors = sorted(divisors)
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    
    # We will iterate M from 13 upwards.
    # To make it efficient, we can jump M or just iterate.
    # The number of valid divisors is ~780.
    # For a given M, the number of candidates is the count of valid_divs <= M.
    
    def can_solve_with_limit(M):
        cands = [d for d in valid_divs if d <= M]
        n = len(cands)
        
        if n == 0:
            return None
            
        # Precompute suffix products? No, dynamic.
        # DFS
        def _dfs(start_idx, current_prod, current_factors):
            if time.time() - start_time > DEADLINE:
                return None
                
            if current_prod == target:
                if 1 <= len(current_factors) <= 12:
                    return list(current_factors)
                return None
                
            if current_prod > target:
                return None
                
            if len(current_factors) > 12:
                return None
                
            # Pruning: 
            # Max product we can still achieve with remaining factors
            # We can pick at most 12 - len(current_factors) factors.
            # The largest available factors are at the end of cands.
            # But we must pick from start_idx onwards.
            # The largest factors in cands[start_idx:] are cands[-1], cands[-2], ...
            
            max_factors_left = 12 - len(current_factors)
            if max_factors_left <= 0:
                return None
                
            # If we pick the largest 'max_factors_left' factors from cands[start_idx:],
            # is their product * current_prod >= target?
            # If not, we can't reach target.
            
            # The largest factors are cands[n-1], cands[n-2], etc.
            # But we must ensure they are >= cands[start_idx].
            # Since cands is sorted, the largest elements in cands[start_idx:] are simply the last elements of cands.
            # However, we can only pick distinct factors.
            
            # Let's just bound by the largest possible product.
            # If current_prod * (cands[-1] ** max_factors_left) < target, prune.
            # This is a weak bound but cheap.
            # A tighter bound: product of the largest 'max_factors_left' elements in cands[start_idx:].
            
            # If n - start_idx < max_factors_left, we can't pick that many.
            # But we don't HAVE to pick that many. We can pick fewer.
            # So the max product is with the largest available count, which is min(max_factors_left, n - start_idx).
            
            available_count = n - start_idx
            if available_count == 0:
                return None
                
            pick_count = min(max_factors_left, available_count)
            
            # Product of the last 'pick_count' elements in cands
            max_prod = 1
            for j in range(pick_count):
                max_prod *= cands[n - 1 - j]
                
            if current_prod * max_prod < target:
                return None

            # Also, if current_prod * (cands[start_idx] ** pick_count) > target?
            # No, we can pick larger ones.
            
            # Another prune: if target % current_prod != 0, prune. (Always true by construction)
            
            for i in range(start_idx, n):
                d = cands[i]
                
                if current_prod * d > target:
                    break
                    
                if time.time() - start_time > DEADLINE:
                    return None

                res = _dfs(i + 1, current_prod * d, current_factors + [d])
                if res:
                    return res
                    
            return None
        
        return _dfs(0, 1, [])

    # Iterate M
    # Start from a reasonable lower bound.
    # (12!)^(1/12) ~ 29.
    # Let's start from 13 to be safe, but it will be fast.
    
    # To speed up, we can binary search if we are sure a solution exists for high M.
    # But linear scan from 13 is fine if the solution is small.
    # If the solution is large, linear scan is slow.
    # Let's use a hybrid: check specific M values or binary search.
    # Actually, the set of valid M is discrete.
    # Let's just binary search on the range [13, target].
    
    lo = 13
    hi = target
    
    # Check if hi works (it always does, e.g., [target] if target > 12 and distinct? 
    # Wait, if k=1, factor is target. target > 12. Distinct? Yes, 1 factor.
    # So hi is always a valid solution.
    
    while lo < hi:
        mid = (lo + hi) // 2
        sol = can_solve_with_limit(mid)
        if sol:
            hi = mid
        else:
            lo = mid + 1
            
    final_sol = can_solve_with_limit(lo)
    
    if final_sol:
        return final_sol
    else:
        return [target]
