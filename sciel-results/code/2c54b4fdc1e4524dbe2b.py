def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Greedy initial solution for good initial bound
    greedy_sum = Fraction(0)
    greedy_indices = []
    for i, d in enumerate(denominators):
        if greedy_sum + Fraction(1, d) < 1:
            greedy_indices.append(i)
            greedy_sum += Fraction(1, d)
    
    # Precompute suffix sums for pruning
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + Fraction(1, denominators[i])
    
    best_sum = greedy_sum
    best_indices = list(greedy_indices)
    
    def backtrack(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if current_sum > best_sum and current_sum < 1:
            best_sum = current_sum
            best_indices = list(current_indices)
        
        if idx == n:
            return
        
        # Prune: even taking all remaining can't beat best
        if current_sum + suffix[idx] <= best_sum:
            return
        
        # Try including denominators[idx]
        frac = Fraction(1, denominators[idx])
        new_sum = current_sum + frac
        if new_sum < 1:
            current_indices.append(idx)
            backtrack(idx + 1, new_sum, current_indices)
            current_indices.pop()
        
        # Try excluding denominators[idx]
        backtrack(idx + 1, current_sum, current_indices)
    
    backtrack(0, Fraction(0), [])
    return best_indices
