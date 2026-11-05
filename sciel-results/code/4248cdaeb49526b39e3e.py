"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    
    # We need distinct factors all > 12, product = target, minimize max factor
    # Binary search on M = maximum allowed factor
    
    # Lower bound: at least one factor > 12, and product must be target
    # If all factors are <= M, and we need distinct ones > 12
    # The product of the smallest k distinct integers > 12 is 13*14*...*12+k
    # We need this <= target for feasibility
    
    # Upper bound: target itself (single factor)
    lo = 13
    hi = target
    
    best_solution = None
    
    # First, try to find any solution quickly with a greedy/DFS approach
    # Then binary search to minimize M
    
    def dfs_find_factors(remaining, min_factor, max_factor, current_factors, depth):
        """Find distinct factors in [min_factor, max_factor] with product = remaining."""
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if remaining == 1:
                return list(current_factors)
            return None
        
        if remaining == 1:
            return list(current_factors)
        
        if len(current_factors) >= 12:
            return None
        
        # We need at least one more factor
        # If remaining > max_factor, we can't use a single factor, need more
        # But each additional factor must be > min_factor and <= max_factor
        # and distinct from current_factors
        
        # Pruning: remaining must be factorable by numbers in [min_factor, max_factor]
        # not in current_factors
        
        # Try factors from max_factor down to min_factor (larger first to reduce depth)
        for f in range(min(max_factor, remaining), min_factor - 1, -1):
            if remaining % f == 0:
                if f in current_factors:
                    continue
                # Check: remaining/f must be factorable into (depth-1) distinct factors
                # all in [min_factor, max_factor], not including f
                # Also remaining/f >= product of smallest (depth-1) distinct ints > 12
                # (excluding f if f would be among them, but f > 12 so this is tricky)
                
                # Simple bound: remaining/f must be >= 1 (trivially true)
                # and remaining/f must be <= product of largest (depth-1) distinct ints <= max_factor
                # not in current_factors ∪ {f}
                
                new_factors = current_factors + [f]
                result = dfs_find_factors(remaining // f, min_factor, max_factor, new_factors, depth - 1)
                if result is not None:
                    return result
        
        return None
    
    def can_form_with_max(remaining, max_factor, min_factor=13):
        """Check if remaining can be written as product of 1-12 distinct integers in [min_factor, max_factor]."""
        if time.time() > deadline:
            return None
        
        if remaining == 1:
            return []
        
        if remaining < min_factor:
            return None
        
        # Try to build the factorization
        # Use DFS: try factors from max_factor down
        current = []
        
        def dfs(rem, min_f, max_f, cur):
            if time.time() > deadline:
                return None
            if rem == 1:
                return list(cur)
            if len(cur) >= 12:
                return None
            if rem < min_f:
                return None
            
            # Upper bound check: rem must be <= product of largest (12-len(cur)) distinct ints <= max_f
            # not in cur. For simplicity, just check rem <= max_f ** (12 - len(cur))
            # which is very loose. Better: check if rem can be divided by at least one valid factor
            
            # Try factors from max_f down to min_f
            # Optimization: if rem is prime and rem > max_f, fail
            # If rem is prime and rem >= min_f and rem <= max_f and rem not in cur, use it
            # and then rem//rem = 1, done
            
            # Try each possible factor
            for f in range(min(max_f, rem), min_f - 1, -1):
                if rem % f == 0:
                    if f in cur:
                        continue
                    new_cur = cur + [f]
                    result = dfs(rem // f, min_f, max_f, new_cur)
                    if result is not None:
                        return result
            return None
        
        return dfs(remaining, min_factor, max_factor, [])
    
    # Binary search on M
    # For a given M, we need to check if target can be factored into distinct ints in [13, M]
    
    # Lower bound refinement: 
    # If we use k factors, the minimum possible product is 13*14*...*12+k
    # The maximum possible product with k factors all <= M is M*(M-1)*...*(M-k+1)
    # We need 13*14*...*12+k <= target <= M*(M-1)*...*(M-k+1)
    
    # Let's first find the minimum M by trying increasing M values
    # But binary search is faster
    
    # Check feasibility for a given M
    def feasible(M):
        """Return factorization if target can be written as product of distinct ints in [13, M], else None."""
        if time.time() > deadline:
            return None
        return can_form_with_max(target, M, 13)
    
    # Binary search
    lo = 13
    hi = target
    best_M = None
    best_factors = None
    
    # First check if hi is feasible (it should be: [target] if target > 12)
    if target > 12:
        # Single factor works
        best_factors = [target]
        best_M = target
    
    # Now binary search for minimum M
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = feasible(mid)
        if result is not None:
            best_factors = result
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_factors is None:
        return [target]
    
    return best_factors

# EVOLVE-BLOCK-END
