"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, log
from functools import lru_cache
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # We need distinct factors all > 12, product = target, minimize max factor
    # Strategy: Binary search on M = maximum allowed factor
    # For each M, check if target can be factored into distinct ints in [13, M]
    
    # Precompute divisors of target that are > 12
    # target = 2^10 * 3^5 * 5^2 * 7 * 11
    # Number of divisors = 11 * 6 * 3 * 2 * 2 = 792
    
    def get_divisors_gt_12(n):
        """Get all divisors of n that are > 12, sorted."""
        divs = set()
        i = 13
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        if n > 12:
            divs.add(n)
        return sorted([d for d in divs if d > 12])
    
    all_divs = get_divisors_gt_12(target)
    div_set = set(all_divs)
    
    def can_form_with_max(remaining, max_factor, min_factor=13, max_depth=12):
        """Check if remaining can be written as product of 1-12 distinct integers in [min_factor, max_factor].
        Returns list of factors or None."""
        if time.time() > deadline:
            return None
        
        if remaining == 1:
            return []
        
        if remaining < min_factor:
            return None
        
        if max_depth <= 0:
            return None
        
        # If remaining itself is a valid factor
        if min_factor <= remaining <= max_factor and remaining in div_set:
            return [remaining]
        
        # Try to split remaining into two parts
        # We need to find distinct factors f1, f2, ... such that product = remaining
        # Use DFS with pruning
        
        # Key optimization: try larger factors first to reduce depth
        # Also, we can use the fact that if we pick factor f, remaining/f must also be factorable
        
        # Precompute: for each possible first factor f in [min_factor, min(max_factor, remaining)],
        # check if remaining % f == 0 and f in div_set and f not used
        
        # Use memoization for (remaining, max_factor, min_factor, used_set) but used_set is too large
        # Instead, use a simpler approach: DFS with strong pruning
        
        def dfs(rem, min_f, max_f, cur_factors, depth_left):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return list(cur_factors)
            
            if depth_left <= 0:
                return None
            
            if rem < min_f:
                return None
            
            # Pruning: if rem > max_f ** depth_left, we can't reach it
            # (loose bound but helps)
            if rem > max_f ** depth_left:
                return None
            
            # Pruning: rem must be >= min_f (already checked)
            # Also, the minimum product of (depth_left) distinct factors >= min_f
            # is min_f * (min_f+1) * ... * (min_f + depth_left - 1)
            # But we can use fewer factors, so this is not a hard lower bound on rem
            # Actually, we need at least 1 factor, and that factor must be <= max_f
            # So rem must be <= max_f (if using 1 factor) or larger if using more
            
            # Try factors from max_f down to min_f
            # But we can limit: f must divide rem, f must be in div_set, f not in cur_factors
            # And rem/f must be factorable into (depth_left - 1) distinct factors in [min_f, max_f]
            
            # Optimization: iterate over divisors of rem that are in [min_f, max_f] and in div_set
            # Sort them in descending order to try larger factors first
            
            # Get candidate factors
            candidates = []
            for f in range(min(max_f, rem), min_f - 1, -1):
                if rem % f == 0 and f in div_set and f not in cur_factors:
                    candidates.append(f)
                    # Don't break; we want to try all, but in descending order
                    # Actually, we should try them in descending order for efficiency
                    # But iterating range in descending order already does this
            
            # Try each candidate
            for f in candidates:
                new_cur = cur_factors | {f}
                result = dfs(rem // f, min_f, max_f, new_cur, depth_left - 1)
                if result is not None:
                    return result
            
            return None
        
        return dfs(remaining, min_factor, max_factor, set(), max_depth)
    
    # Binary search on M
    lo = 13
    hi = target
    best_factors = None
    best_M = target
    
    # Single factor always works
    best_factors = [target]
    
    # Binary search for minimum M
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = can_form_with_max(target, mid, 13, 12)
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
