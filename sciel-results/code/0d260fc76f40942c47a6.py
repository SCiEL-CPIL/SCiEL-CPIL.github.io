def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    # Suffix sums for pruning
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + fractions[i]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Greedy initialization
    current_sum = Fraction(0)
    current_indices = []
    for i in range(n):
        if current_sum + fractions[i] < 1:
            current_sum += fractions[i]
            current_indices.append(i)
            
    if current_sum > best_sum:
        best_sum = current_sum
        best_indices = list(current_indices)
        
    # Branch and Bound
    # Use a list for current_indices to avoid copying
    curr_indices = []
    
    def dfs(idx, current_sum):
        nonlocal best_sum, best_indices
        
        # Pruning 1: Sum exceeded 1
        if current_sum >= 1:
            return
            
        # Pruning 2: Even taking all remaining, we can't beat best_sum
        if current_sum + suffix[idx] <= best_sum:
            return
            
        # Base case
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(curr_indices)
            return
        
        # Option 1: Take fractions[idx]
        curr_indices.append(idx)
        dfs(idx + 1, current_sum + fractions[idx])
        curr_indices.pop()
        
        # Option 2: Skip fractions[idx]
        dfs(idx + 1, current_sum)
        
    dfs(0, Fraction(0))
    
    return best_indices
