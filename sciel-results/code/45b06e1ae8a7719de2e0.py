"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    target = math.factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (90s limit)
    
    # Precompute all divisors of target that are > 12
    # target = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = []
    for i in range(13, target + 1):
        if target % i == 0:
            divisors.append(i)
    
    # Sort divisors
    divisors.sort()
    
    # Create a set for O(1) lookup
    divisor_set = set(divisors)
    
    # We'll do iterative deepening on the maximum factor
    # Binary search on M: can we partition target into distinct divisors all in (12, M]?
    
    best_solution = [target]
    best_max = target
    
    # DFS: remaining product, last chosen factor (for ordering), current factors list
    # We pick factors in increasing order to ensure distinctness
    
    def can_partition(remaining, min_factor, max_factor, current_factors, depth):
        """
        Try to partition 'remaining' into distinct factors, each in [min_factor, max_factor],
        each a divisor of target, and the product equals remaining.
        Returns True if possible, and fills current_factors.
        """
        if time.time() > deadline:
            return False
        
        if remaining == 1:
            return len(current_factors) >= 1
        
        if depth >= 12:
            return False
        
        # If remaining itself is a valid factor (>= min_factor, <= max_factor, divisor)
        # and we can stop here
        if remaining >= min_factor and remaining <= max_factor and remaining in divisor_set:
            # Check distinctness
            if remaining not in current_factors:
                current_factors.append(remaining)
                if len(current_factors) >= 1:
                    result = True
                    current_factors.pop()
                    return True
                current_factors.pop()
        
        # Try factors from min_factor to min(max_factor, remaining)
        # But we need them to be divisors of 'remaining' (since remaining/factor must be integer)
        # And divisors of target
        
        # Get divisors of remaining that are in [min_factor, min(max_factor, remaining)]
        # and are also divisors of target
        start = min_factor
        end = min(max_factor, remaining)
        
        for f in range(start, end + 1):
            if time.time() > deadline:
                return False
            
            if remaining % f != 0:
                continue
            if f not in divisor_set:
                continue
            if f in current_factors:
                continue
            
            new_remaining = remaining // f
            # The next factors must be > f (for distinctness, since we're going in order)
            # Actually no - we just need them to be distinct. But to avoid permutations,
            # we enforce increasing order.
            
            current_factors.append(f)
            if can_partition(new_remaining, f + 1, max_factor, current_factors, depth + 1):
                return True
            current_factors.pop()
        
        return False
    
    # Binary search on the maximum factor
    lo = 13
    hi = target
    
    # First, let's try to find a good solution with a reasonable upper bound
    # Let's try M values from small to large
    
    # Actually, let's do a more direct search: DFS with pruning, trying to minimize max
    # We'll use the fact that we want to minimize the max, so we try smaller max values first
    
    # Let's try specific M values
    # Lower bound: at least 77 (7*11) if they must be in same factor, or lower if split
    # Actually 7 and 11 can be in different factors, so lower bound is lower
    
    # Let me try a smarter approach: iterative deepening DFS
    # For each candidate max M (from low to high), check if feasible
    
    # Precompute divisors of target > 12
    # For each M, the candidate factors are divisors of target in (12, M]
    
    # Let's try M from 20 upward
    found = False
    for M in range(20, min(target, 200000)):
        if time.time() > deadline:
            break
        
        # Check if M is a divisor of target (it doesn't have to be, but factors must be)
        # Get all divisors of target in (12, M]
        candidates = [d for d in divisors if d <= M]
        
        if len(candidates) < 1:
            continue
        
        # DFS: pick a subset of candidates whose product is target
        # Using backtracking with pruning
        
        def dfs(remaining, start_idx, current_factors):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                if len(current_factors) >= 1 and len(current_factors) <= 12:
                    return list(current_factors)
                return None
            
            if len(current_factors) >= 12:
                return None
            
            # Pruning: product of all remaining candidates from start_idx onward
            # must be >= remaining
            # Simple pruning: if remaining > 1 and no more candidates, fail
            
            # Try each candidate from start_idx
            for i in range(start_idx, len(candidates)):
                if time.time() > deadline:
                    return None
                
                f = candidates[i]
                if f > remaining:
                    break
                if remaining % f != 0:
                    continue
                
                new_remaining = remaining // f
                current_factors.append(f)
                
                result = dfs(new_remaining, i + 1, current_factors)
                if result is not None:
                    return result
                
                current_factors.pop()
            
            return None
        
        current_factors = []
        result = dfs(target, 0, current_factors)
        
        if result is not None:
            best_solution = result
            best_max = M
            found = True
            break
    
    if not found:
        # Fallback: return [target]
        best_solution = [target]
        best_max = target
    
    return best_solution
# EVOLVE-BLOCK-END
