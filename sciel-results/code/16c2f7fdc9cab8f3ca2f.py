"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        """Approximate the n-th harmonic number."""
        if n <= 0:
            return 0
        if n == 1:
            return 1.0
        return math.log(n) + gamma + 1.0/(2.0*n) - 1.0/(12.0*n*n)
    
    def can_complete(remainder, min_denom, max_denom):
        """
        Check if it's possible to sum to 'remainder' using distinct denominators 
        in [min_denom, max_denom].
        Uses harmonic sum upper bound for pruning.
        """
        if remainder == 0:
            return True
        if remainder < 0:
            return False
        
        # If min_denom > max_denom, impossible
        if min_denom > max_denom:
            return False
        
        # Max possible sum is sum(1/d for d in range(min_denom, max_denom+1))
        # Approx H_max - H_{min_denom-1}
        if min_denom <= 1:
            max_sum = harmonic_approx(max_denom)
        else:
            max_sum = harmonic_approx(max_denom) - harmonic_approx(min_denom - 1)
        
        # If even the max possible sum is less than remainder, impossible
        if max_sum < float(remainder) * 0.999:  # Small tolerance
            return False
        
        # Minimum number of terms needed: each term <= 1/min_denom
        # So we need at least ceil(remainder * min_denom) terms
        # But we can only use (max_denom - min_denom + 1) terms
        min_terms_needed = math.ceil(float(remainder) * min_denom)
        available_terms = max_denom - min_denom + 1
        if min_terms_needed > available_terms:
            return False
        
        return True
    
    def dfs(remainder, min_denom, current_set, max_denom, depth, visited):
        """
        DFS to find a valid Egyptian fraction expansion.
        Returns True if a solution is found.
        """
        nonlocal best_solution, best_max_denom
        
        if time.time() - start_time > time_limit:
            return False
            
        if remainder == 0:
            # Found a solution
            actual_max = max(current_set) if current_set else 0
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
        if remainder > 0:
            lower_bound = (remainder.denominator + remainder.numerator - 1) // remainder.numerator
        else:
            lower_bound = min_denom
        
        if lower_bound < min_denom:
            lower_bound = min_denom
            
        # Upper bound is max_denom
        if lower_bound > max_denom:
            return False
            
        # Limit depth to avoid excessive recursion
        if depth > 256:
            return False
            
        # Iterate d from lower_bound to max_denom
        # Try denominators in order that might lead to smaller max
        for d in range(lower_bound, max_denom + 1):
            if time.time() - start_time > time_limit:
                return False
                
            # Skip if d is already used
            if d in visited:
                continue
                
            next_rem = remainder - Fraction(1, d)
            
            if next_rem < 0:
                # 1/d > remainder, try larger d
                continue
                
            if next_rem == 0:
                # Solution found
                new_set = current_set + [d]
                actual_max = max(new_set)
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = list(new_set)
                return True
            
            # Recurse with updated state
            visited.add(d)
            if dfs(next_rem, d + 1, current_set + [d], max_denom, depth + 1, visited):
                visited.remove(d)
                return True
            visited.remove(d)
                
        return False
    
    # Strategy: Binary search on the maximum denominator bound
    # We know a solution exists for K=1736
    # We want to find the minimum K such that a solution exists
    
    low = 7  # Minimum possible (1/d < 5/31 means d > 31/5 = 6.2)
    high = best_max_denom
    
    # First, try to find a better solution with a tighter bound
    # Do a few iterations of binary search
    
    # To speed up, we can try some heuristic bounds first
    # Try K = 100, 200, 500, 1000 to see if solutions exist
    
    test_bounds = [100, 200, 300, 500, 800, 1200, 1500]
    feasible_bounds = []
    
    for K in test_bounds:
        if K > high:
            continue
        if time.time() - start_time > time_limit:
            break
        
        # Reset for this test
        test_visited = set()
        result = dfs(target, low, [], K, 0, test_visited)
        
        if result:
            feasible_bounds.append(K)
            # Update best if we found something better
            if best_max_denom <= K:
                break
    
    # Now do binary search in the feasible range
    if feasible_bounds:
        # Find the smallest feasible K
        search_low = feasible_bounds[0]
        search_high = high
        
        while search_low < search_high and time.time() - start_time < time_limit:
            mid = (search_low + search_high) // 2
            test_visited = set()
            result = dfs(target, low, [], mid, 0, test_visited)
            
            if result:
                search_high = mid
            else:
                search_low = mid + 1
        
        # Try the final bound
        test_visited = set()
        dfs(target, low, [], search_low, 0, test_visited)
    
    # If we still have the original solution or found a better one, return it
    if best_solution is not None and len(best_solution) > 0 and len(best_solution) <= 256:
        # Verify the solution
        total = sum(Fraction(1, d) for d in best_solution)
        if total == target and all(d >= 1 and d <= 10000000 for d in best_solution):
            if len(set(best_solution)) == len(best_solution):
                return best_solution
    
    # Fallback: return the known good solution
    return [8, 28, 1736]
# EVOLVE-BLOCK-END
