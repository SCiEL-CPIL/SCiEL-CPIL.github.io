from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    target = Fraction(5, 31)
    
    # Helper to check if a set of denominators is valid
    def is_valid(denoms):
        if not denoms: return False
        if len(denoms) > 256: return False
        if len(set(denoms)) != len(denoms): return False
        if any(d < 1 or d > 10000000 for d in denoms): return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target

    # Helper to solve for a given max denominator D
    # Returns a list of denominators or None
    def solve_for_D(D):
        # We search for any subset of {1, ..., D} summing to target
        # Using DFS
        path = []
        
        # Precompute reciprocals?
        # Pruning: 
        # 1. Sum of 1/d for d in path + sum of 1/d for d in (min_d ... D) >= target?
        # Actually, we just need to reach target.
        
        def dfs(rem, min_d):
            if time.time() > deadline:
                return None
            if rem == 0:
                return list(path)
            if len(path) >= 256:
                return None
            
            # Max possible sum we can add from min_d to D
            # If rem is very small, we need large d.
            # If rem is large, we need small d.
            
            # Lower bound for next d: 1/d <= rem => d >= 1/rem
            # Also d >= min_d
            d_min = max(min_d, (1 / rem).numerator // (1 / rem).denominator + 1) # ceil(1/rem)
            # Wait, Fraction arithmetic is better.
            # d >= 1/rem => rem >= 1/d => rem*d >= 1 => d >= 1/rem
            if rem > 0:
                d_min = max(min_d, (1 / rem).numerator // (1 / rem).denominator + 1)
            else:
                return None # rem should not be negative in valid path
            
            # Upper bound for next d:
            # Even if we take all remaining denominators from d to D, can we reach rem?
            # Sum_{i=d}^D 1/i >= rem
            # This is a strong prune.
            # Also, we can't take more than 256 - len(path) terms.
            # But we can take any number.
            
            # Iterate d from d_min to D
            for d in range(d_min, D + 1):
                if rem < Fraction(1, d):
                    continue
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break # Since d increases, 1/d decreases, so new_rem will only increase? 
                    # No, if 1/d > rem, we skip. If 1/d <= rem, new_rem >= 0.
                    # As d increases, 1/d decreases, so new_rem increases.
                    # So if new_rem < 0, it means 1/d > rem. We should have skipped.
                    # Actually, the loop condition `if rem < 1/d: continue` handles this.
                
                path.append(d)
                res = dfs(new_rem, d + 1)
                if res:
                    return res
                path.pop()
                
            return None

        return dfs(target, 1)

    # Binary search on D
    lo = 1
    hi = 1736 # Known solution max denom
    best_sol = [7, 56, 1736] # Fallback
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = solve_for_D(mid)
        if sol:
            best_sol = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    return best_sol
