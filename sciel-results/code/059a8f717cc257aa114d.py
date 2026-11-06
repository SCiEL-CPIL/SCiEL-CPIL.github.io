import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # Precompute harmonic numbers approximation?
    # H_n = ln(n) + gamma + 1/(2n)
    # We can use this for pruning.
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        if n <= 0:
            return 0
        return math.log(n) + gamma + 1.0/(2.0*n)
        
    def can_complete(remainder, min_denom, max_denom):
        """
        Check if it's possible to sum to 'remainder' using distinct denominators 
        in (min_denom, max_denom].
        Uses harmonic sum upper bound.
        """
        if remainder <= 0:
            return remainder == 0
        
        # Max possible sum is sum(1/d for d in range(min_denom+1, max_denom+1))
        # Approx H_max - H_min
        if max_denom <= min_denom:
            return False
            
        max_sum = harmonic_approx(max_denom) - harmonic_approx(min_denom)
        
        # If even the max possible sum is less than remainder, impossible
        if max_sum < float(remainder):
            return False
            
        # Also, if the smallest possible term 1/(min_denom+1) is too big?
        # No, we can pick larger denominators.
        # But if 1/min_denom > remainder, we can't use min_denom.
        # The loop handles this.
        
        return True

    def dfs(remainder, min_denom, current_set, max_denom, depth):
        """
        Returns True if a solution is found, updating best_solution if max_denom is smaller.
        Actually, for the feasibility check, we just need to know if ANY solution exists with max <= max_denom.
        """
        if time.time() - start_time > time_limit:
            return False
            
        if remainder == 0:
            # Found a solution
            # The max denom in this solution is max(current_set)
            # But we are searching within bound 'max_denom'.
            # We want to record the actual max denom of this specific solution.
            actual_max = max(current_set)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = list(current_set)
            return True
            
        if remainder < 0:
            return False
            
        # Pruning: Can we complete?
        if not can_complete(remainder, min_denom, max_denom):
            return False
            
        # Lower bound for next denominator
        # 1/d <= remainder => d >= 1/remainder
        lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
        if lower_bound < min_denom + 1:
            lower_bound = min_denom + 1
            
        # Upper bound is max_denom
        if lower_bound > max_denom:
            return False
            
        # Iterate d
        # Heuristic: Try smaller d first?
        # Smaller d => larger term => smaller remainder.
        # This is usually good for finding solutions quickly.
        
        for d in range(lower_bound, max_denom + 1):
            if time.time() - start_time > time_limit:
                return False
                
            next_rem = remainder - Fraction(1, d)
            
            if next_rem < 0:
                # 1/d > remainder
                # Since d is increasing, 1/d is decreasing.
                # So next_rem will increase.
                # We should continue, not break.
                continue
                
            if next_rem == 0:
                # Solution found
                actual_max = max(current_set + [d])
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = list(current_set) + [d]
                return True
                
            # Recurse
            # Next min denom is d
            if dfs(next_rem, d, current_set + [d], max_denom, depth + 1):
                return True
                
        return False

    # Strategy:
    # We know a solution exists for K=1736.
    # We want to find the minimum K.
    # We will binary search K in [1, 1736].
    
    low = 1
    high = best_max_denom
    
    # We already have a solution for high.
    # We need to check if a solution exists for smaller K.
    
    # To avoid re-running the full DFS for K=1736 (which we know works),
    # we start checking from high-1 downwards? 
    # Binary search is better if the feasibility function is monotonic.
    # If a solution exists for K, does it exist for K+1? 
    # Yes, because the set of valid denominators for K is a subset of those for K+1.
    # So if feasible(K) is True, feasible(K+1) is True.
    # We want the smallest K such that feasible(K) is True.
    
    # However, the DFS might be slow for large K if no solution is found (infeasible).
    # But we are searching for the *minimum* K.
    # If we check K=1000 and it's infeasible, we know K must be > 1000.
    # If we check K=1736 and it's feasible, we know K <= 1736.
    
    # Let's do binary search.
    # But we must be careful: the DFS has a time limit.
    # If it times out, we don't know if it's feasible or not.
    # We should treat timeout as "Infeasible" for the purpose of narrowing down?
    # No, if it times out, it might be feasible.
    # Better: Run the DFS for a specific K.
    # If it finds a solution, update best.
    # If it exhausts search without finding, it's infeasible for that K.
    
    # Optimization:
    # Instead of binary search which might hit expensive infeasible cases,
    # we can try to improve the current best by stepping down.
    # But binary search is O(log N).
    
    # Let's try a few specific values first to get a better upper bound?
    # The seed found 1736.
    # Let's try K = 1000.
    # If feasible, high = 1000.
    # If not, low = 1001.
    
    # Actually, let's just use the binary search range [1, 1736].
    
    # To speed up the "Infeasible" determination, the pruning is key.
    
    # One issue: `dfs` modifies `best_solution` and `best_max_denom` non-locally.
    # For the binary search logic, we just need a boolean: "Does a solution exist with max <= K?"
    # We can write a separate function `find_solution_with_max(k)` that returns the solution or None.
    
    def find_solution_with_max(k):
        """
        Attempts to find a solution with all denominators <= k.
        Returns the list of denominators if found, else None.
        """
        # Reset time check? No, global time limit applies.
        
        def dfs_local(remainder, min_denom, current_set):
            if time.time() - start_time > time_limit:
                return None
                
            if remainder == 0:
                return current_set
                
            if remainder < 0:
                return None
                
            # Pruning
            if not can_complete(remainder, min_denom, k):
                return None
                
            lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
            if lower_bound < min_denom + 1:
                lower_bound = min_denom + 1
                
            if lower_bound > k:
                return None
                
            for d in range(lower_bound, k + 1):
                if time.time() - start_time > time_limit:
                    return None
                    
                next_rem = remainder - Fraction(1, d)
                
                if next_rem < 0:
                    continue
                    
                if next_rem == 0:
                    return current_set + [d]
                    
                # Recurse
                res = dfs_local(next_rem, d, current_set + [d])
                if res is not None:
                    return res
                    
            return None
            
        return dfs_local(target, 0, [])

    # Binary Search for minimum K
    # We know K=1736 works.
    # We want to find min K.
    
    # To save time, we can check a few midpoints.
    # If time is running out, return best_solution.
    
    # Note: The seed program's DFS was slightly different (it iterated up to best_max).
    # My `find_solution_with_max` is stricter (iterates up to k).
    
    # Let's run the binary search.
    low = 1
    high = best_max_denom
    
    # We already have a solution for high.
    # We don't need to re-verify high if we trust the seed.
    # But let's assume we need to find the min.
    
    # To avoid expensive "infeasible" searches that take the whole time,
    # we can add a small per-node time check or just rely on the global limit.
    
    # If we find a solution for K, we update best_solution and best_max_denom.
    # Then we set high = K - 1.
    
    # If we determine K is infeasible (search exhausted), we set low = K + 1.
    
    # How to distinguish "Exhausted (Infeasible)" from "Timeout (Unknown)"?
    # `find_solution_with_max` returns None in both cases.
    # If it's a timeout, we don't know.
    # Strategy:
    # 1. Try to find a solution for K.
    # 2. If found, update best, high = K-1.
    # 3. If not found:
    #    a. If time is nearly up, break.
    #    b. Assume infeasible? Risky.
    #    c. Maybe try a smaller step?
    
    # Given the 60s limit, and the fact that 1736 is not huge,
    # the search space for K=1736 is manageable if pruning is good.
    # For smaller K, the space is smaller.
    # Infeasible cases for small K are fast to prove.
    # Infeasible cases for large K (close to 1736) might be slow if no solution exists.
    # But we know a solution exists for 1736.
    # So the "Infeasible" cases are for K < 1736.
    # As K decreases, the space shrinks.
    
    # Let's try linear decrease from 1736?
    # 1735, 1734...
    # If 1735 is infeasible, it might be fast.
    # If 1000 is infeasible, it might be fast.
    # Binary search is safer.
    
    # Let's use Binary Search, but handle timeouts by breaking.
    
    while low <= high:
        if time.time() - start_time > time_limit - 2.0:
            break
            
        mid = (low + high) // 2
        
        # Try to find solution with max <= mid
        sol = find_solution_with_max(mid)
        
        if sol is not None:
            # Found a solution
            max_in_sol = max(sol)
            if max_in_sol < best_max_denom:
                best_max_denom = max_in_sol
                best_solution = sol
                
            # A solution exists for mid, so one exists for any K >= mid.
            # We want smaller K.
            high = mid - 1
        else:
            # No solution found.
            # Could be infeasible or timeout.
            # If timeout, we should stop to avoid wasting time.
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Assume infeasible for this mid.
            low = mid + 1
            
    return best_solution
