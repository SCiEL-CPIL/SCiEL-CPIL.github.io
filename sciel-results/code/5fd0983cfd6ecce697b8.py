"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    n = 12
    target = factorial(n)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1
    
    # Find all divisors of target
    divisors = set()
    for i in range(1, isqrt(target) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # For search: we need distinct factors > 12, product = target
    # Minimize the largest factor
    # Use DFS with branch and bound
    
    best_max = target  # initial: single factor
    best_solution = [target]
    
    # Sort divisors for systematic search
    # We'll build factors in increasing order to ensure distinctness
    
    def dfs(remaining, min_factor, chosen, current_max):
        nonlocal best_max, best_solution, start_time, DEADLINE
        
        if time.time() > DEADLINE:
            return
        
        if remaining == 1:
            if len(chosen) >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(chosen)
            return
        
        # Pruning: even if we use the smallest possible remaining factors,
        # can we still reach product = remaining with distinct factors > min_factor?
        # Lower bound on max: remaining must be achievable
        
        # Upper bound check: if remaining > 1, the smallest factor we can pick
        # is at least min_factor (or 13 if min_factor <= 12)
        # The maximum factor will be at least remaining / (product of smallest possible factors)
        
        # Try each valid divisor d of remaining where d >= min_factor and d > 12
        # and d <= remaining (obviously)
        # Also d must be such that remaining/d can be factored into distinct factors > d
        
        # Optimization: d should be a divisor of remaining
        # And d >= min_factor to ensure distinctness (since we pick in increasing order)
        
        # Also, we can have at most 12 - len(chosen) more factors
        # And the current_max is max(chosen) if chosen else 0
        
        # Branch and bound: if current_max >= best_max, we can still improve
        # but if even the theoretical minimum max >= best_max, prune
        
        # Theoretical minimum: if we pick k more factors, the smallest possible
        # max is at least (remaining)^(1/k) for the remaining factors
        # But this is a weak bound. Let's use: the next factor d must satisfy
        # d * (d+1) * ... * (d+k-1) <= remaining (very weak)
        
        # Better: just enumerate divisors of remaining that are >= min_factor
        # and > 12
        
        for d in valid_divisors:
            if d < min_factor:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            # Pruning: if new_max >= best_max, we might still be ok if we can
            # achieve a smaller max later, but since we're building in order,
            # the max will only increase or stay same
            # Actually, the max of the final solution is max(all chosen factors)
            # Since we're picking in increasing order, the last factor will be the max
            # So if d >= best_max, and we still need more factors, the final max >= d >= best_max
            # But if this is the last factor (new_remaining == 1), then max = new_max
            if new_max >= best_max and new_remaining != 1:
                # Even if we pick the smallest possible next factor, the max won't decrease
                # But actually the max is determined by the largest factor overall
                # If d is already >= best_max, and we need more factors, the final max >= d
                # So we can't improve. But wait - we could have a smaller max if we
                # don't pick d now. But we're enumerating in order, so this is just one branch.
                # Actually this pruning is valid: if d >= best_max and new_remaining > 1,
                # then any completion will have max >= d >= best_max, so no improvement.
                continue
            
            if new_remaining == 1:
                if new_max < best_max:
                    best_max = new_max
                    best_solution = list(chosen) + [d]
            else:
                # Check: can new_remaining be factored into distinct factors all > d?
                # Minimum number of factors: 1 (just new_remaining itself, if > d)
                # Maximum number of factors: limited by how many distinct divisors > d
                # Quick check: new_remaining > d (so it could be a single factor)
                if new_remaining > d:
                    chosen.append(d)
                    dfs(new_remaining, d + 1, chosen, new_max)
                    chosen.pop()
                # Also check if new_remaining can be split further
                # But if new_remaining <= d, it can't be a valid next factor
                # (must be > d for distinctness in increasing order)
                # So only proceed if new_remaining > d
        
        # Also consider: maybe we should stop and use remaining as the last factor
        # But that's handled when d == remaining (new_remaining == 1)
    
    dfs(target, 13, [], 0)
    
    if time.time() > DEADLINE:
        return best_solution
    
    # Verify solution
    prod = 1
    for x in best_solution:
        prod *= x
    assert prod == target, f"Product mismatch: {prod} != {target}"
    assert len(set(best_solution)) == len(best_solution), "Factors not distinct"
    assert all(x > 12 for x in best_solution), "Factor not > 12"
    assert 1 <= len(best_solution) <= 12, "Wrong number of factors"
    
    return best_solution
# EVOLVE-BLOCK-END
