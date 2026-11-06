"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    start_time = time.time()
    deadline = 55.0 # Safety margin for 60s limit

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Weights are 1/denominator
    # We want to maximize sum of weights subject to sum < 1
    # Sort by weight descending (denominator ascending) for better B&B bounds
    # Index i in sorted list corresponds to original index
    items = []
    for i, d in enumerate(denominators):
        items.append((1.0 / d, i))
        
    # Sort by weight descending
    items.sort(key=lambda x: -x[0])
    
    # Extract sorted weights and original indices
    sorted_weights = [w for w, _ in items]
    sorted_orig_indices = [idx for _, idx in items]
    
    # Precompute suffix sums of weights for quick pruning
    # suffix_sum[i] = sum of sorted_weights[i:]
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]
        
    best_sum = 0.0
    best_indices = []
    
    # Branch and Bound
    # idx: current index in sorted list
    # curr_sum: current sum of selected weights
    # selected: list of original indices selected so far
    def dfs(idx, curr_sum, selected):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
            
        # If we've processed all items
        if idx == n:
            if curr_sum > best_sum and curr_sum < 1.0:
                best_sum = curr_sum
                best_indices = list(selected)
            return
            
        # Pruning 1: Even if we take all remaining items, can we beat best?
        # Note: We must be strictly below 1. 
        # If curr_sum + suffix_sum[idx] <= best_sum, we can't beat it.
        if curr_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Pruning 2: Fractional Knapsack Upper Bound
        # Calculate the maximum possible sum by taking items fractionally
        cap = 1.0 - curr_sum
        bound_sum = curr_sum
        for i in range(idx, n):
            if cap <= 0:
                break
            if sorted_weights[i] <= cap:
                bound_sum += sorted_weights[i]
                cap -= sorted_weights[i]
            else:
                bound_sum += cap
                cap = 0
                break
                
        if bound_sum <= best_sum:
            return
            
        # Branch 1: Include item idx
        # Check if adding this item keeps us strictly below 1
        if curr_sum + sorted_weights[idx] < 1.0:
            selected.append(sorted_orig_indices[idx])
            dfs(idx + 1, curr_sum + sorted_weights[idx], selected)
            selected.pop()
            
        # Branch 2: Exclude item idx
        dfs(idx + 1, curr_sum, selected)
        
    # Initial call
    dfs(0, 0.0, [])
    
    # If no solution found (should not happen as 1/2 < 1), fallback to greedy
    if not best_indices:
        # Greedy fallback
        curr_sum = 0.0
        for w, idx in items:
            if curr_sum + w < 1.0:
                curr_sum += w
                best_indices.append(idx)
                best_sum = curr_sum
                
    # Verify with Fraction to be absolutely sure about strict inequality and optimality check
    # The B&B uses floats which might have precision issues near 1.
    # Let's verify the best candidate found.
    # Also, it's possible that due to float precision, we missed a slightly better rational sum?
    # Unlikely for N=47, but let's do a final local search with Fractions on the best found set
    # to ensure we are strictly below 1 and try to improve if float precision was an issue.
    
    # Actually, the B&B explores all valid subsets that could potentially beat the current best.
    # The main risk is if `curr_sum + w < 1.0` is True in float but False in Fraction (i.e. sum is exactly 1 or slightly above due to rounding error in float addition).
    # Since weights are 1/d, the sum is a rational number.
    # If the true sum is 1, float sum might be 0.9999999999999999 or 1.0000000000000002.
    # If float says < 1, but Fraction says == 1, we have an invalid solution.
    # We must verify the final answer with Fraction.
    
    final_sum_frac = sum(Fraction(1, denominators[i]) for i in best_indices)
    
    if final_sum_frac >= 1:
        # Invalid solution found by float logic.
        # Remove the last added item? Or restart with stricter float check?
        # Better: Perform a small local search with Fractions to find a valid neighbor.
        # Or just use the greedy solution which is safer?
        # Let's try to fix the current best by removing items until valid, then trying to add others.
        # But this might be suboptimal.
        
        # Safer approach: If float verification fails, re-run B&B with a stricter epsilon or use Fraction in B&B?
        # Given N=47, Fraction B&B might be too slow.
        # Let's assume float precision is sufficient for N=47 with weights 1/d.
        # The sum of 47 terms of 1/d has a denominator that is a multiple of LCM(2..48).
        # The error in float addition is bounded.
        # To be safe, let's verify and if invalid, fallback to a known valid greedy set.
        
        # Greedy set is always valid if we check with Fraction?
        # Greedy: add if curr_sum + w < 1.
        # If we use Fraction in greedy, it's exact.
        
        # Let's just re-compute a greedy solution with Fraction as a safe fallback.
        curr_sum_frac = Fraction(0)
        fallback_indices = []
        for i in range(n):
            d = denominators[i]
            w = Fraction(1, d)
            if curr_sum_frac + w < 1:
                curr_sum_frac += w
                fallback_indices.append(i)
        
        if curr_sum_frac > final_sum_frac:
             best_indices = fallback_indices
             final_sum_frac = curr_sum_frac
        else:
             # If the B&B solution was invalid, we must use the fallback
             best_indices = fallback_indices
             final_sum_frac = curr_sum_frac

    # One more check: Can we improve the fallback/B&B result with a quick Fraction-based local search?
    # The B&B should have found the global optimum in the float space.
    # If the float space optimum was valid, it's likely the global rational optimum.
    
    return best_indices
# EVOLVE-BLOCK-END
