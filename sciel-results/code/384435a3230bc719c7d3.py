"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 60s limit
    
    def verify(denoms):
        """Verify that the sum of reciprocals of denoms equals 1."""
        total = sum(Fraction(1, d) for d in denoms)
        return total == 1
    
    # Pre-computed known solution: 16 terms
    known_solution = [5, 6, 7, 10, 14, 21, 30, 35, 42, 70, 105, 140, 210, 280, 420, 840]
    if verify(known_solution):
        best = known_solution
    else:
        best = None
    
    # Try to find a solution with fewer terms using a smarter search
    # First, try very small sets (2-5 terms) with a bounded denominator range
    # The denominators must be >= 5 and distinct
    
    def search_small_sets(max_terms, max_denom=200):
        """Try to find a small set of denominators that sums to 1."""
        if time.time() > deadline:
            return None
        
        # For small sets, we can use a more targeted approach
        # The smallest denominator must be at least 5
        # For k terms, the sum of 1/d_i = 1
        
        # Try all combinations of small denominators
        candidates = list(range(5, max_denom + 1))
        
        for k in range(2, max_terms + 1):
            if time.time() > deadline:
                return None
            
            # Use a pruning strategy: the sum of the k smallest possible reciprocals
            # starting from some minimum must be able to reach 1
            # Actually, let's just try combinations but with pruning
            
            # For efficiency, let's use a recursive approach with pruning
            def dfs(remaining, min_denom, used, depth, target_depth):
                if time.time() > deadline:
                    return None
                if depth == target_depth:
                    if remaining == 0:
                        return sorted(used)
                    return None
                if remaining == 0:
                    return sorted(used)
                
                # The next denominator must be at least ceil(1/remaining)
                if remaining <= 0:
                    return None
                min_d = max(min_denom, math.ceil(1 / remaining))
                
                # We need (target_depth - depth) more terms
                # Each term is at most 1/min_d, so remaining <= (target_depth - depth) / min_d
                if remaining > (target_depth - depth) / min_d:
                    return None
                
                # Also, the sum of the largest possible (target_depth - depth) terms
                # starting from min_d must be at least remaining
                # The largest terms would be 1/min_d, 1/(min_d+1), ...
                # But for a quick check, (target_depth - depth) * (1/min_d) >= remaining is necessary
                # This is already covered above
                
                # Try denominators from min_d up to some reasonable limit
                # The maximum denominator is bounded by the fact that 1/d < remaining
                # so d > 1/remaining
                max_d = min(max_denom, int(1 / remaining) + 100) if remaining > 0 else max_denom
                # Actually, 1/d < remaining means d > 1/remaining, but we want 1/d <= remaining
                # so d >= ceil(1/remaining) which is min_d
                # The upper bound: if we use d, the remaining after that must be achievable
                # with (target_depth - depth - 1) terms each at most 1/(d+1)
                # So remaining - 1/d <= (target_depth - depth - 1) / (d+1)
                # This gives us a tighter upper bound
                
                for d in range(min_d, max_d + 1):
                    if d < 5:
                        continue
                    if d in used:
                        continue
                    new_remaining = remaining - Fraction(1, d)
                    if new_remaining < 0:
                        break
                    result = dfs(new_remaining, d + 1, used | {d}, depth + 1, target_depth)
                    if result is not None:
                        return result
                return None
            
            # Start DFS for this target depth
            result = dfs(Fraction(1, 1), 5, set(), 0, k)
            if result is not None:
                return result
        
        return None
    
    # Try to find solutions with fewer than 16 terms
    # Start with small numbers of terms and increase
    for max_terms in range(2, 16):
        if time.time() > deadline:
            break
        
        # Use a larger denominator range for larger term counts
        if max_terms <= 5:
            max_denom = 500
        elif max_terms <= 8:
            max_denom = 2000
        else:
            max_denom = 10000
        
        result = search_small_sets(max_terms, max_denom)
        if result is not None:
            if verify(result):
                if best is None or len(result) < len(best):
                    best = result
                # If we found a very small solution, we can stop
                if len(result) <= 5:
                    return result
    
    # If we found a better solution, return it
    if best is not None:
        return sorted(best)
    
    # Fall back to the known solution
    return known_solution

# EVOLVE-BLOCK-END
