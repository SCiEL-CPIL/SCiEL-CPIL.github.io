def construct():
    from fractions import Fraction
    import sys

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Sort indices by value descending (denominator ascending)
    indices = list(range(n))
    indices.sort(key=lambda i: -values[i])
    
    sorted_values = [values[i] for i in indices]
    sorted_original_indices = indices
    
    # Precompute suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_values[i]
        
    best_sum = Fraction(0)
    best_indices = []
    current_indices = []
    
    sys.setrecursionlimit(10000)
    
    def dfs(idx, current_sum):
        nonlocal best_sum, best_indices
        
        if current_sum >= 1:
            return
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = current_indices[:]
            
        if idx == n:
            return
            
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Include
        if current_sum + sorted_values[idx] < 1:
            current_indices.append(sorted_original_indices[idx])
            dfs(idx + 1, current_sum + sorted_values[idx])
            current_indices.pop()
            
        # Exclude
        dfs(idx + 1, current_sum)
        
    dfs(0, Fraction(0))
    
    return sorted(best_indices)
