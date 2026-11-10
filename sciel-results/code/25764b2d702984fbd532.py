import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Known good solution with max 248
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    best_solution = [8, 31, 248]
    best_max = 248
    
    def _dfs(rem, min_d, max_d, used, depth, max_depth):
        if time.time() > deadline:
            return None
        if rem == 0:
            return []
        if depth >= max_depth:
            return None
        if rem < 0:
            return None
            
        # Lower bound for next denominator
        # 1/d < rem => d > 1/rem
        d_min = max(min_d, (rem.denominator // rem.numerator) + 1)
        if rem.numerator * d_min <= rem.denominator:
            d_min += 1
            
        if d_min > max_d:
            return None
            
        k = max_depth - depth
        
        # Pruning: Max possible sum
        # Sum of 1/d for the k smallest available denominators >= d_min
        # We approximate by just taking the next k integers, ignoring 'used' for speed
        # If this sum is less than rem, we can't reach rem.
        s_max = 0
        d_check = d_min
        count = 0
        while count < k:
            if d_check > max_d:
                break
            s_max += Fraction(1, d_check)
            count += 1
            d_check += 1
        if s_max < rem:
            return None
            
        # Pruning: Min possible sum
        # Sum of 1/d for the k largest possible denominators <= max_d
        s_min = 0
        d_check = max_d
        count = 0
        while count < k:
            if d_check < 1:
                break
            s_min += Fraction(1, d_check)
            count += 1
            d_check -= 1
        if s_min > rem:
            return None
            
        for d in range(d_min, max_d + 1):
            if time.time() > deadline:
                return None
            if d in used:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
            used.add(d)
            res = _dfs(new_rem, d + 1, max_d, used, depth + 1, max_depth)
            used.discard(d)
            if res is not None:
                return [d] + res
        return None

    # Try to find solutions with smaller max denominator
    # We iterate downwards from best_max - 1
    # But we can just call DFS with max_d = best_max - 1
    # If it finds a solution, update best_max and repeat.
    
    current_limit = best_max - 1
    
    while current_limit > 10:
        if time.time() > deadline - 2:
            break
            
        # Try to find a solution with all denominators <= current_limit
        # Start with empty used, min_d = 1 (but effectively 7 due to target)
        res = _dfs(target, 1, current_limit, set(), 0, 10)
        
        if res is not None:
            sol = res
            # Verify
            if len(set(sol)) == len(sol) and all(1 <= d <= 10000000 for d in sol):
                s = sum(Fraction(1, d) for d in sol)
                if s == target:
                    max_d = max(sol)
                    if max_d < best_max:
                        best_max = max_d
                        best_solution = sol
                        current_limit = max_d - 1
                    else:
                        # Found a solution but not better? Should not happen if limit is correct
                        break
                else:
                    break
            else:
                break
        else:
            # No solution found with this limit. 
            # It might be that no solution exists with max <= current_limit.
            # We can stop here if we assume monotonicity? 
            # If no solution exists with max <= M, does a solution exist with max > M?
            # Yes, we already have one with 248.
            # So if no solution with max <= 247, then 248 is optimal?
            # Not necessarily. There could be a solution with max 240, but the DFS missed it?
            # The DFS is exhaustive for the given limit.
            # So if it returns None, no solution exists with all d <= current_limit.
            # Thus, the minimal max denominator is > current_limit.
            # Since we have a solution with best_max, and no solution with max <= best_max - 1,
            # best_max is the minimum.
            break

    return best_solution
