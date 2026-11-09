def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 85
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Suffix sums
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + reciprocals[i]
    
    best_deficit = Fraction(1)
    best_indices = []
    
    def backtrack(idx, current_sum, current_indices):
        nonlocal best_deficit, best_indices
        
        if time.time() > deadline:
            return
        
        if current_sum >= 1:
            return
        
        # Check if current is better
        if current_indices:
            deficit = 1 - current_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_indices = current_indices[:]
        
        if idx >= n:
            return
        
        # If we can add all remaining and stay < 1
        if current_sum + suffix_sum[idx] < 1:
            # Best in this branch: take all remaining
            new_sum = current_sum + suffix_sum[idx]
            new_indices = current_indices + list(range(idx, n))
            deficit = 1 - new_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_indices = new_indices
            return
        
        # Can't take all, need to branch
        # Skip
        backtrack(idx + 1, current_sum, current_indices)
        
        # Include
        new_sum = current_sum + reciprocals[idx]
        if new_sum < 1:
            backtrack(idx + 1, new_sum, current_indices + [idx])
    
    backtrack(0, Fraction(0), [])
    
    return best_indices
