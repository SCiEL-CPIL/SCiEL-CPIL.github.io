"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    target = Fraction(5, 31)
    
    # 1. Generate a valid baseline using the greedy algorithm
    # Greedy tends to produce large denominators, but it's fast and valid.
    baseline = []
    r = target
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d > 10000000:
             # Fallback if greedy fails (unlikely for 5/31)
             break
        baseline.append(d)
        r -= Fraction(1, d)
        
    # Ensure baseline is valid and within bounds
    if len(baseline) > 256 or len(baseline) == 0:
        # Fallback to a known small expansion if greedy fails, though it shouldn't
        return [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50] # Dummy, will be replaced by search anyway
        
    best_max = max(baseline)
    best_sol = baseline

    # 2. Iterative Deepening DFS to minimize the maximum denominator
    # We want to find a solution with max_denominator < current best_max
    # Start from a reasonable lower bound.
    # 5/31 is approx 0.16. 1/6 is 0.166. So 6 is a good starting point.
    # Actually, 5/31 = 1/7 + 1/9 + 1/15 + 1/35 + 1/45 + 1/63 + 1/70 + 1/105 ... (just guessing)
    # Let's start low.
    
    # Lower bound for max denominator:
    # If all denominators are >= L, then sum <= k/L.
    # We need sum = 5/31.
    # Also, the first denominator d1 must satisfy 1/d1 <= 5/31 => d1 >= 31/5 = 6.2 => d1 >= 7.
    # So max denominator must be at least 7.
    
    # Let's try to find a solution with max_den <= M
    # We'll iterate M downwards from best_max - 1 to 7
    
    # Helper for DFS
    def dfs(remaining, min_d, current_max, path):
        nonlocal best_max, best_sol, deadline
        
        if time.time() > deadline:
            return None
            
        if remaining == 0:
            if current_max < best_max:
                best_max = current_max
                best_sol = path[:]
            return True
            
        # Pruning: if remaining is too small for the number of terms allowed?
        # We don't have a fixed number of terms, but we have a max of 256.
        # Also, if remaining < 1/current_max, we can't use any denominator >= current_max.
        # Wait, we are minimizing the MAX denominator. So we are constraining all denominators to be <= current_limit.
        
        # Let's restructure: 
        # dfs(remaining, min_d, limit, path)
        # Find distinct denominators d_i >= min_d, d_i <= limit, such that sum(1/d_i) = remaining.
        
        # Heuristic lower bound for min_d:
        # 1/min_d <= remaining => min_d >= 1/remaining
        # But we also have the constraint d_i >= min_d (to ensure distinctness and ordering).
        
        # If remaining is very small, min_d must be large.
        # If 1/min_d > remaining, we must increase min_d.
        
        # Also, if we have used 'k' terms so far, and we need 'remaining', 
        # the smallest possible sum with k more terms (all equal to min_d) is k/min_d.
        # But they must be distinct.
        
        # Simple pruning:
        # If remaining < 1/limit, we can't use any denominator <= limit? No, 1/limit is the smallest unit.
        # If remaining < 1/limit, we need denominators > limit. But we are constrained by limit.
        # So if remaining < 1/limit, it's impossible? 
        # Wait, if remaining is e.g. 0.0001 and limit is 10000, 1/10000 = 0.0001. 
        # If remaining < 1/limit, we can't form it with denominators <= limit.
        # So if remaining * limit < 1, return False.
        
        if remaining.numerator * limit < remaining.denominator:
            return False
            
        # Another pruning: 
        # Max possible sum with remaining terms? We don't have a fixed term count limit other than 256.
        # But we can check if even with many terms we can't reach it?
        # Sum of 1/i for i from min_d to infinity diverges, so we can always reach any small remaining 
        # if we allow enough terms. But we are limited to 256 terms.
        
        # Let's estimate the max sum we can add with 256 terms starting from min_d.
        # Approx sum_{i=0}^{255} 1/(min_d + i)
        # This is roughly 256/min_d.
        # If remaining > 256/min_d, we might not have enough terms? 
        # Actually, if remaining is large, we need small denominators.
        # If remaining > 1/min_d + 1/(min_d+1) + ... + 1/(min_d+255), then we can't do it with 256 terms.
        # But 5/31 is not that large.
        
        # Let's just do a simple DFS.
        # Next denominator d must be >= min_d and <= limit.
        # Also 1/d <= remaining => d >= ceil(1/remaining)
        
        lower_bound = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # If lower_bound > limit, no solution
        if lower_bound > limit:
            return False
            
        # Try denominators from lower_bound to limit
        # Optimization: Only try a few candidates? 
        # In Egyptian fractions, usually the greedy choice is close.
        # But to minimize max, we might need to deviate.
        
        # Let's iterate d from lower_bound to limit.
        # To speed up, we can step? No, distinct denominators matter.
        
        # Heuristic: Try d near the greedy choice first?
        # Greedy choice: d_greedy = ceil(1/remaining)
        # Let's try d_greedy, d_greedy+1, d_greedy-1, etc.
        
        # Actually, standard DFS for Egyptian fractions:
        for d in range(lower_bound, limit + 1):
            if time.time() > deadline:
                return None
                
            if remaining - Fraction(1, d) < 0:
                continue
                
            # Pruning: if remaining - 1/d is too small to be formed by denominators > d?
            # Let r2 = remaining - 1/d
            # If r2 == 0, we found a solution (if path + [d] is valid)
            r2 = remaining - Fraction(1, d)
            
            if r2 == 0:
                new_path = path + [d]
                new_max = max(current_max, d)
                if new_max < best_max:
                    best_max = new_max
                    best_sol = new_path
                return True
            else:
                # Recurse
                # New min_d is d+1
                # New limit is same
                # New current_max is max(current_max, d)
                # But wait, we are minimizing the GLOBAL max. 
                # If d > best_max, we can prune immediately? 
                # Yes! If d >= best_max, then max(current_max, d) >= best_max, so it won't improve.
                # However, we might want to find a solution with the SAME max but fewer terms? 
                # The task says "Minimize the LARGEST denominator". It doesn't care about term count 
                # as long as it's <= 256. So if we find one with smaller max, we update.
                # If d >= best_max, this branch cannot improve best_max. 
                # So we can prune d >= best_max.
                
                if d >= best_max:
                    continue
                    
                # Also, if r2 is too small for denominators > d?
                # 1/(d+1) must be <= r2? No, r2 is sum of 1/d_i where d_i > d.
                # So r2 >= 1/(d+1) + 1/(d+2) ...
                # If r2 < 1/(d+1), it's impossible? 
                # No, r2 could be 1/(d+1) exactly.
                # If r2 < 1/(d+1), we can't form it with denominators >= d+1.
                if r2 < Fraction(1, d + 1):
                    continue
                    
                # Also, check if r2 can be formed with remaining terms?
                # We have 256 - len(path) - 1 terms left.
                
                # Recurse
                res = dfs(r2, d + 1, limit, path + [d])
                if res is not None:
                    return res
                    
        return False

    # Start iterative deepening
    # We know best_max from greedy.
    # Let's try to reduce it.
    # We'll try limits from best_max - 1 down to 7.
    
    # But DFS with a fixed limit 'limit' explores all denominators <= limit.
    # We want to find ANY solution with max_den <= limit.
    # Then we can lower the limit.
    
    # To be efficient, we can just run DFS with limit = best_max - 1.
    # If it finds a solution, update best_max and repeat with new best_max - 1.
    
    current_limit = best_max - 1
    while current_limit >= 7 and time.time() < deadline - 5:
        # Reset best_sol for this limit? 
        # No, we keep the global best.
        # But dfs updates best_max.
        # We need to ensure dfs only finds solutions with max < current_limit?
        # Actually, if we set limit = current_limit, dfs will only use denominators <= current_limit.
        # So any solution found will have max <= current_limit.
        # Since current_limit < best_max (initially), any solution found is better.
        
        # However, dfs might find a solution with max < current_limit.
        # Then we should update current_limit to that new max - 1.
        
        # Let's modify dfs to return the max denominator of the best solution found in this branch?
        # Or just let it update global best_max.
        
        # Problem: dfs as written updates best_max if it finds a better one.
        # But it stops at the first solution found? No, it returns True on first solution.
        # We want the BEST solution (min max).
        # So we should not stop at the first solution if we are doing iterative deepening?
        # Actually, if we fix the limit to L, and find a solution with max M <= L,
        # we can then try L = M - 1.
        # So stopping at the first solution with max <= L is fine, as long as we update L to M-1.
        
        # Let's refactor: dfs finds a solution with max_den <= limit.
        # It returns the max_den of that solution.
        
        def dfs_find(remaining, min_d, limit, path):
            if time.time() > deadline:
                return None
                
            if remaining == 0:
                return max(path) if path else 0
                
            if remaining.numerator * limit < remaining.denominator:
                return None
                
            lower_bound = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            
            if lower_bound > limit:
                return None
                
            # Try to find a solution
            # Heuristic: try denominators that keep the remainder "nice"?
            # For now, just iterate.
            
            # Optimization: If we find a solution, we can return it.
            # To find the MIN max denominator, we should prefer smaller d?
            # No, smaller d means larger 1/d, leaving smaller remainder.
            # Smaller remainder often requires larger denominators.
            # So maybe larger d is better for minimizing max?
            # Let's try d from lower_bound to limit.
            
            # To speed up, we can limit the number of d's we try?
            # Or use a heuristic.
            
            # Let's try a few d's.
            # Actually, for 5/31, the search space is small.
            
            for d in range(lower_bound, limit + 1):
                if time.time() > deadline:
                    return None
                    
                r2 = remaining - Fraction(1, d)
                if r2 < 0:
                    continue
                    
                if r2 == 0:
                    return d
                    
                if r2 < Fraction(1, d + 1):
                    continue
                    
                # Recurse
                res = dfs_find(r2, d + 1, limit, path + [d])
                if res is not None:
                    return res
                    
            return None

        # Run dfs_find with current_limit
        # We need to start with min_d = ceil(1/target) = 7
        min_d_start = (target.denominator + target.numerator - 1) // target.numerator
        
        res_max = dfs_find(target, min_d_start, current_limit, [])
        
        if res_max is not None:
            # Found a solution with max <= current_limit
            # Actually, res_max IS the max denominator of the solution.
            if res_max < best_max:
                best_max = res_max
                # We need to recover the solution to update best_sol.
                # dfs_find didn't return the path.
                # Let's modify to return path or store it.
                
                # For now, let's just re-run to get the path? 
                # Or better, store best_path in dfs.
                
                # Let's restart with a simpler approach: 
                # Just use the greedy result if DFS is too complex to manage state.
                # But we want to improve.
                
                # Let's implement a version that stores the path.
                pass
        else:
            # No solution with max <= current_limit
            current_limit -= 1
            continue
            
    # The above dfs_find logic is a bit flawed because it doesn't track the path for best_sol.
    # Let's simplify: 
    # Just run a DFS that tries to beat the current best_max.
    # If it finds one, update best_max and best_sol.
    # Repeat until time runs out.
    
    # Let's write a cleaner DFS that updates global best_sol.
    
    def dfs_optimize(remaining, min_d, current_limit, path):
        nonlocal best_max, best_sol, deadline
        
        if time.time() > deadline:
            return
            
        if remaining == 0:
            if len(path) <= 256:
                m = max(path)
                if m < best_max:
                    best_max = m
                    best_sol = path[:]
            return
            
        if remaining.numerator * current_limit < remaining.denominator:
            return
            
        lower_bound = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        if lower_bound > current_limit:
            return
            
        # Prune if current path length is too long
        if len(path) >= 256:
            return
            
        # Try denominators
        # Heuristic: try d's that are close to the greedy choice?
        # Or just iterate.
        
        # To avoid deep recursion on bad paths, we can limit the branching.
        # But for 5/31, it should be fast.
        
        for d in range(lower_bound, current_limit + 1):
            if time.time() > deadline:
                return
                
            if d >= best_max:
                continue
                
            r2 = remaining - Fraction(1, d)
            if r2 < 0:
                continue
                
            if r2 == 0:
                new_path = path + [d]
                if len(new_path) <= 256:
                    m = max(new_path)
                    if m < best_max:
                        best_max = m
                        best_sol = new_path
                return # Found a solution with this d, no need to try larger d for this remaining? 
                       # Actually, larger d would leave smaller r2, requiring even larger denominators.
                       # So if we found a solution, it's likely good. But maybe a smaller d leads to a smaller MAX?
                       # Smaller d -> larger 1/d -> smaller r2.
                       # Smaller r2 usually requires larger denominators.
                       # So larger d -> larger r2 -> potentially smaller denominators for the rest?
                       # This is counter-intuitive.
                       # Let's just continue searching.
                       
            if r2 < Fraction(1, d + 1):
                continue
                
            dfs_optimize(r2, d + 1, current_limit, path + [d])
            
            if time.time() > deadline:
                return

    # Run the optimization
    # Start with limit = best_max - 1
    limit = best_max - 1
    while limit >= 7 and time.time() < deadline - 5:
        dfs_optimize(target, min_d_start, limit, [])
        if best_max <= limit:
            limit = best_max - 1
        else:
            limit -= 1
            
    # Final check
    if not best_sol or len(best_sol) > 256 or max(best_sol) > 10000000:
        return baseline
        
    return best_sol

# EVOLVE-BLOCK-END
