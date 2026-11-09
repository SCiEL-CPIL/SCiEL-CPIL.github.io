"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """Find Egyptian fraction of 5/31 minimizing the largest denominator.
    
    Searches for the smallest M such that there exist distinct integers
    in [7, M] whose reciprocals sum to exactly 5/31.
    """
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    target = Fraction(5, 31)
    
    # Known fallback solution
    fallback = [8, 31, 248]
    
    # Try increasing max denominator M starting from 7
    # d1 must be >= ceil(31/5) = 7
    for M in range(7, 10000001):
        if time.time() > deadline:
            break
        
        # Try to find subset of {7, ..., M} summing to target
        # Use DFS: pick denominators in increasing order
        # State: (start_idx, remaining_sum, chosen_list)
        
        def dfs(start, remaining, chosen):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                return chosen[:]
            
            if start > M:
                return None
            
            # Pruning: if sum of all reciprocals from start to M is less than remaining
            # Compute max possible sum from start to M
            # This is expensive to compute exactly each time, so use a simpler bound
            # The sum of 1/i from i=start to M is at most (M - start + 1) / start
            # But more tightly, it's at most 1/start + 1/(start+1) + ... 
            # For pruning, if even taking all remaining gives less than needed, prune
            # Quick upper bound: number of remaining terms * (1/start)
            remaining_count = M - start + 1
            if remaining_count * Fraction(1, start) < remaining:
                return None
            
            # Try including each denominator from start to M
            for d in range(start, M + 1):
                if time.time() > deadline:
                    return None
                
                term = Fraction(1, d)
                if term > remaining:
                    continue  # Too large, skip (but since d increases, all further are smaller, so we can't break)
                
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                
                chosen.append(d)
                result = dfs(d + 1, new_remaining, chosen)
                if result is not None:
                    return result
                chosen.pop()
            
            return None
        
        result = dfs(7, target, [])
        if result is not None:
            # Verify the solution
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(1 <= d <= 10000000 for d in result) and len(result) <= 256:
                return result
    
    return fallback
# EVOLVE-BLOCK-END
