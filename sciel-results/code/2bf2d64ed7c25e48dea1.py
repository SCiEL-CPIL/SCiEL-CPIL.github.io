# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Create list of (value, original_index)
    items = [(Fraction(1, d), i) for i, d in enumerate(denominators)]
    # Sort by value descending (largest fractions first)
    items.sort(key=lambda x: -x[0])
    
    # Precompute suffix sums for pruning
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + items[i][0]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Initial heuristic: Greedy
    curr_sum = Fraction(0)
    curr_indices = []
    for i in range(n):
        val, orig_idx = items[i]
        if curr_sum + val < 1:
            curr_sum += val
            curr_indices.append(orig_idx)
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_indices = curr_indices[:]

    # Backtracking
    # path stores original indices
    def dfs(idx, current_sum, path):
        nonlocal best_sum, best_indices
        
        # Prune if max possible sum is not better than best
        if current_sum + suffix[idx] <= best_sum:
            return
        
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = path[:]
            return
        
        val, orig_idx = items[idx]
        
        # Try including the item
        new_sum = current_sum + val
        if new_sum < 1:
            path.append(orig_idx)
            dfs(idx + 1, new_sum, path)
            path.pop()
            
        # Try excluding the item
        dfs(idx + 1, current_sum, path)

    dfs(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
