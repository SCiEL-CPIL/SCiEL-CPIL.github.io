def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    fractions = [Fraction(1, d) for d in denominators]
    N = len(fractions)
    
    suffix_sum = [Fraction(0) for _ in range(N + 1)]
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + fractions[i]
        
    best_sum = Fraction(0)
    best_indices = []
    current_indices = []
    
    def dfs(i, current_sum):
        nonlocal best_sum, best_indices, current_indices
        
        # Pruning
        if current_sum + suffix_sum[i] <= best_sum:
            return
            
        if i == N:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
            
        # Option 1: Include fractions[i]
        new_sum = current_sum + fractions[i]
        if new_sum < 1:
            current_indices.append(i)
            dfs(i + 1, new_sum)
            current_indices.pop()
            
        # Option 2: Exclude fractions[i]
        dfs(i + 1, current_sum)

    dfs(0, Fraction(0))
    
    return best_indices
