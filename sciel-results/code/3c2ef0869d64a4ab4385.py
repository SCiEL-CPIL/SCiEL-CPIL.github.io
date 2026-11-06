"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def construct():
    target = math.factorial(12)
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Generate all divisors of target
    def get_divisors(n):
        divs = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(list(divs))
        
    # Filter divisors > 12
    all_divs = [d for d in get_divisors(target) if d > 12]
    
    best_factors = None
    best_max = float('inf')
    
    # Strategy:
    # 1. Try to find solutions with k=5, 6, 7 distinct factors.
    # 2. Use DFS with strong pruning.
    # 3. Since k=5 gave max=96 in previous analysis, and k=6/7 likely have larger max,
    #    we prioritize finding *any* valid solution quickly, then try to minimize max.
    #    Actually, more factors -> smaller geometric mean -> potentially smaller max.
    #    Geometric mean for k=5: (12!)^(1/5) ~ 40.
    #    Geometric mean for k=6: (12!)^(1/6) ~ 30.
    #    Geometric mean for k=7: (12!)^(1/7) ~ 24.
    #    So k=7 *could* have a smaller max than k=5, but the factors must be distinct integers > 12.
    #    Let's search for k=7, 6, 5 in that order, but with a time limit per k or global.
    #    If we find a solution with max M, we can prune any branch where current_max >= M.
    
    # To make DFS fast, we pre-compute divisors and sort them.
    # We also need to ensure distinctness. We do this by iterating in increasing order
    # and only picking divisors larger than the last picked one.
    
    # Optimization: 
    # If we pick k factors, their product is target.
    # Let the factors be d1 < d2 < ... < dk.
    # d1 > 12.
    # dk is the max. We want to minimize dk.
    
    # We can also use the fact that if we fix the first k-1 factors, the last one is determined.
    # So we can search for k-1 factors whose product divides target, and check if the remainder
    # is a valid k-th factor (distinct, >12, > last picked, and < current best_max).
    
    # Let's implement a recursive search for k factors.
    # To speed it up, we can iterate k from 7 down to 5.
    # For each k, we run DFS.
    
    # We'll use a global time check to ensure we return within deadline.
    
    def search(k, remaining, start_idx, last_val, current_factors, current_max):
        nonlocal best_factors, best_max
        
        if time.time() > deadline:
            return
            
        if k == 1:
            # The last factor is 'remaining'
            if remaining > last_val and remaining > 12:
                # Check distinctness: remaining must not be in current_factors
                # Since we enforce increasing order, remaining > last_val >= all previous, so distinctness is guaranteed.
                new_max = max(current_max, remaining)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [remaining]
            return
        
        if k == 0:
            if remaining == 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = current_factors[:]
            return
            
        # Pruning:
        # The minimum possible product of k distinct factors > last_val is roughly (last_val+1)*...*(last_val+k).
        # If remaining is smaller than this, it's impossible.
        # However, calculating this exactly might be slow. 
        # A simpler bound: The smallest k distinct integers > last_val have product P_min.
        # If remaining < P_min, return.
        # P_min = product_{i=1..k} (last_val + i)
        p_min = 1
        for i in range(1, k+1):
            p_min *= (last_val + i)
            if p_min > remaining:
                return
                
        # Also, the largest factor in the remaining k factors must be <= best_max - 1 (to improve).
        # So the product of the remaining k factors must be <= remaining, and each factor <= best_max.
        # This is already handled by the loop bounds below.
        
        # We iterate through possible next factors d.
        # d must be:
        # 1. > last_val
        # 2. divides remaining
        # 3. < best_max (strictly, to improve)
        # 4. index >= start_idx
        
        # To speed up, we can break early if d is too large.
        # The maximum d we can pick is min(best_max - 1, remaining // (product of min possible remaining k-1 factors))
        # But let's just iterate and prune.
        
        # We can limit the upper bound of d.
        # If we pick d, the remaining k-1 factors must multiply to remaining//d.
        # The smallest product of k-1 distinct factors > d is P_min_rem.
        # So remaining//d >= P_min_rem => d <= remaining // P_min_rem.
        
        for i in range(start_idx, len(all_divs)):
            d = all_divs[i]
            
            if d > last_val:
                # Check if d is too large to improve best_max
                if d >= best_max:
                    break
                    
                if remaining % d == 0:
                    new_remaining = remaining // d
                    new_max = max(current_max, d)
                    
                    # Prune if new_max >= best_max
                    if new_max >= best_max:
                        break
                        
                    # Prune if remaining product is too small for k-1 factors
                    # Min product of k-1 distinct factors > d
                    p_min_rem = 1
                    for j in range(1, k):
                        p_min_rem *= (d + j)
                        if p_min_rem > new_remaining:
                            break
                    if p_min_rem > new_remaining:
                        # Since d is increasing, p_min_rem will only increase, so we can break?
                        # No, d increases, so d+j increases, so p_min_rem increases.
                        # But new_remaining decreases as d increases? No, new_remaining = remaining // d.
                        # As d increases, new_remaining decreases.
                        # So the condition p_min_rem > new_remaining might become true and stay true?
                        # Not necessarily, but likely. Let's just continue or break carefully.
                        # Actually, if p_min_rem > new_remaining, it's impossible.
                        # As d increases, p_min_rem increases and new_remaining decreases, so the inequality becomes stronger.
                        # So we can break.
                        break
                        
                    search(k - 1, new_remaining, i + 1, d, current_factors + [d], new_max)
            else:
                continue
                
    # We try k=7, 6, 5.
    # We might find a solution for k=7 with a smaller max than k=5.
    # If we find a solution for k=7, we update best_max. Then when searching k=6, we prune more aggressively.
    
    # To ensure we find *some* solution quickly, we can also try a heuristic search first?
    # Or just let the DFS run.
    
    # Let's run search for k=7, then k=6, then k=5.
    # We stop if time runs out.
    
    for k in [7, 6, 5]:
        search(k, target, 0, 12, [], 0)
        if time.time() > deadline - 5:
            break
            
    if best_factors is None:
        # Fallback: return [target] which is valid (1 factor, >12, product target)
        # Max is target, which is large, but valid.
        return [target]
        
    return best_factors
# EVOLVE-BLOCK-END
