import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Fallback solution
    best_solution = [8, 31, 248]
    
    # We know 1/d < 5/31 => d > 31/5 = 6.2, so d >= 7
    min_denom = 7
    
    # We iterate M from min_denom upwards
    # The optimal M is likely small.
    # If M is large, the search space grows, but the pruning `max_possible_remaining < target` becomes less effective 
    # if the sum of reciprocals from last+1 to M is large.
    # However, for small M, it's very effective.
    
    # Precompute suffix sums of reciprocals for quick max_possible_remaining check?
    # For a fixed M, we can precompute S[i] = sum(1/j for j in range(i, M+1)).
    # Then max_possible_remaining from last_denom is S[last_denom+1].
    
    for M in range(min_denom, 100): # Try up to 100
        if time.time() > deadline:
            break
            
        # Precompute suffix sums for this M
        # suffix[i] = sum(1/j for j in range(i, M+1))
        suffix = [0] * (M + 2)
        for i in range(M, min_denom - 1, -1):
            suffix[i] = suffix[i+1] + Fraction(1, i)
        
        # DFS
        def dfs(last_denom, current_sum):
            if current_sum > target:
                return None
            if current_sum == target:
                return [] # Empty list of additional denoms, handled by caller
                
            # Pruning: Can we reach target with remaining denoms?
            # Remaining denoms are from last_denom+1 to M
            if last_denom + 1 <= M:
                max_add = suffix[last_denom + 1]
            else:
                max_add = Fraction(0)
                
            if current_sum + max_add < target:
                return None
                
            # Try adding each next denom
            for d in range(last_denom + 1, M + 1):
                # Optimization: If 1/d is already too big to add without exceeding target?
                # No, we check current_sum + 1/d > target in the recursive call or here.
                if current_sum + Fraction(1, d) > target:
                    # Since d increases, 1/d decreases. 
                    # Wait, if current_sum + 1/d > target, then for larger d, 1/d is smaller.
                    # So we cannot break. We must continue.
                    # Actually, if we pick a larger d, the sum increases less.
                    # So we just skip this d if it exceeds? No, if it exceeds, larger d might not.
                    # But we are iterating d in increasing order.
                    # If 1/d is small, it might be fine.
                    # So we just check.
                    pass
                
                next_sum = current_sum + Fraction(1, d)
                if next_sum > target:
                    continue
                    
                res = dfs(d, next_sum)
                if res is not None:
                    return [d] + res
                    
            return None
            
        # Start DFS
        # We can start with last_denom = min_denom - 1 = 6
        # So first choice is d in 7..M
        result = dfs(min_denom - 1, Fraction(0))
        
        if result is not None:
            # The result is a list of denominators
            # We need to return them
            return result
            
    return best_solution
