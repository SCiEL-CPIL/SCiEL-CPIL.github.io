"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute suffix sums for upper bound pruning
    # suffix_sum[i] = sum of 1/denominators[j] for j >= i
    suffix_sum = [Fraction(0) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, denominators[i])
    
    best_deficit = None
    best_indices = None
    
    def search(idx, current_sum, selected):
        nonlocal best_deficit, best_indices
        
        if time.time() > DEADLINE:
            return
        
        # Prune: even taking all remaining can't beat best
        if best_deficit is not None:
            max_possible = current_sum + suffix_sum[idx]
            if max_possible <= 1 - best_deficit:
                return
        
        # If we've considered all denominators
        if idx == n:
            if current_sum < 1:
                deficit = 1 - current_sum
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(selected)
            return
        
        # Option 1: Skip current denominator
        search(idx + 1, current_sum, selected)
        
        if time.time() > DEADLINE:
            return
        
        # Option 2: Include current denominator (if sum stays < 1)
        new_sum = current_sum + Fraction(1, denominators[idx])
        if new_sum < 1:
            selected.append(idx)
            search(idx + 1, new_sum, selected)
            selected.pop()
    
    search(0, Fraction(0), [])
    
    if best_indices is None:
        # Fallback: should not happen, but just in case
        best_indices = [0]  # just 1/2, deficit = 1/2
    
    return best_indices
# EVOLVE-BLOCK-END
