"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    target = math.factorial(12)
    deadline = time.time() + 85.0
    
    # Find all divisors of target that are > 12
    divisors = []
    for i in range(13, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other > 12 and other != i:
                divisors.append(other)
    
    # Ensure we have target itself as a potential divisor (though it's > 12, it's the whole product)
    # Actually, we need to be careful. The factors must be > 12.
    # Let's collect all divisors > 12.
    all_divs = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            all_divs.add(i)
            all_divs.add(target // i)
    
    candidates = sorted([d for d in all_divs if d > 12])
    
    best_factors = None
    best_max = float('inf')
    
    # We want to find a subset of distinct candidates from `candidates` that multiply to target.
    # To minimize the largest factor, we should search for solutions with smaller max factors first?
    # Or just find any valid solution and track the min max.
    # Since the number of factors is variable (1 to 12), and distinct.
    
    # A single factor [target] is always a valid solution with max = target.
    best_factors = [target]
    best_max = target
    
    # We can prune: if current_max >= best_max, we can stop exploring this branch if we are trying to minimize max.
    # But we don't know the final max until we pick the last factor.
    # However, if we pick a factor larger than best_max, the final max will be at least that large.
    # So we can prune if any candidate we consider is >= best_max.
    
    # Let's use DFS.
    # State: remaining product, index in candidates to start from (to ensure distinctness and avoid permutations), 
    #        current list of factors, current max.
    
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_factors, best_max
        
        if time.time() > deadline:
            return
            
        if remaining == 1:
            # Found a solution
            if current_max < best_max:
                best_max = current_max
                best_factors = current_factors[:]
            return
            
        # If no more factors can be picked or remaining is 1
        if start_idx >= len(candidates):
            return
            
        # Pruning: 
        # 1. If remaining is in candidates[start_idx:], we can pick it as the last factor.
        # 2. We should iterate candidates in reverse order? Or forward?
        #    To minimize the max, we want to avoid large factors. 
        #    However, if we pick a small factor, we might be left with a large remaining.
        #    Actually, the "largest factor" is the max of the set.
        #    If we pick a factor `f`, the new max is max(current_max, f).
        #    If max(current_max, f) >= best_max, we can prune.
        
        for i in range(start_idx, len(candidates)):
            f = candidates[i]
            
            # If f is greater than or equal to best_max, we can break?
            # No, because candidates are sorted ascending. If f >= best_max, any subsequent f' will also be >= best_max.
            # And since f is part of the product, the max will be at least f.
            # So if f >= best_max, we can break the loop.
            if f >= best_max:
                break
                
            if remaining % f == 0:
                new_remaining = remaining // f
                new_max = max(current_max, f)
                
                # If new_max >= best_max, we can skip this branch entirely?
                # Yes, because any completion will have max >= new_max >= best_max.
                if new_max < best_max:
                    current_factors.append(f)
                    dfs(new_remaining, i + 1, current_factors, new_max)
                    current_factors.pop()
                    
                # Optimization: if new_remaining == 1, we already handled it in the base case?
                # No, the base case is at the start of the function.
                # So if new_remaining == 1, the recursive call will hit the base case and update best.
                
                # Additional pruning: if new_remaining > 1 and new_remaining < candidates[i+1] (if exists)
                # and new_remaining is not a candidate >= i+1, then we can't complete?
                # Actually, if new_remaining is not divisible by any candidate from i+1 onwards, we fail.
                # We can check if new_remaining has any divisor in candidates[i+1:] that is < best_max.
                # This is expensive. Let's stick to the basic pruning.
                
                # Another pruning: if new_remaining == 1, we are done.
                # If new_remaining > 1, we must be able to form it from remaining candidates.
                
                # Let's add a check: if new_remaining is in candidates[i+1:], we can potentially finish in 1 step.
                # If not, we need more steps.
                
                # Heuristic: If new_remaining is small, it might be hard to form from large candidates.
                # But candidates are > 12.
                
                # If new_remaining < candidates[i+1] (if i+1 < len), then no single factor can equal it,
                # and any product of factors > 12 will be > new_remaining (if new_remaining < 12*12=144? No).
                # Actually, if new_remaining < candidates[i+1], it cannot be formed by a single factor.
                # Can it be formed by multiple? 
                # The smallest product of two factors > 12 is 13*14 = 182.
                # So if new_remaining < 182, it cannot be formed by 2 or more factors.
                # Thus, if new_remaining < 182 and new_remaining != 1, it must be a single factor.
                # So if new_remaining < 182 and new_remaining is not in candidates[i+1:], we can prune.
                
                if new_remaining > 1:
                    if new_remaining < 182:
                        # Must be a single factor
                        if new_remaining not in candidates[i+1:]:
                            continue
                    else:
                        # Could be single or multiple.
                        # If new_remaining is not in candidates[i+1:], we need at least 2 factors.
                        # The smallest product of 2 distinct factors > 12 is 13*14 = 182.
                        # If new_remaining < 182, it can't be 2 factors.
                        # If new_remaining >= 182, it might be 2 factors.
                        pass

    dfs(target, 0, [], 0)
    
    return best_factors
# EVOLVE-BLOCK-END
