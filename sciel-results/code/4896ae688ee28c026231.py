def construct():
    from fractions import Fraction
    import sys
    
    # Increase recursion limit
    sys.setrecursionlimit(10000)

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Sort fractions in descending order to maximize pruning effectiveness
    # We store (value, original_index)
    indexed_values = sorted([(v, i) for i, v in enumerate(values)], key=lambda x: x[0], reverse=True)
    sorted_values = [v for v, i in indexed_values]
    sorted_orig_indices = [i for v, i in indexed_values]
    
    # Precompute suffix sums: sum of sorted_values[i:]
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_values[i]
        
    # Initialize best solution with a greedy approach
    # Greedy: take items in descending order if they fit
    best_sum = Fraction(0)
    best_mask = 0
    current_sum = Fraction(0)
    for i in range(n):
        if current_sum + sorted_values[i] < 1:
            current_sum += sorted_values[i]
            best_mask |= (1 << i)
    best_sum = current_sum
    
    # Branch and Bound search
    # idx: current index in sorted_values
    # curr_sum: sum of selected values so far
    # mask: bitmask of selected values in sorted order
    def search(idx, curr_sum, mask):
        nonlocal best_sum, best_mask
        
        # Base case
        if idx == n:
            if curr_sum < 1:
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_mask = mask
            return
            
        # Pruning: If even taking all remaining items doesn't beat best_sum
        if curr_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Option 1: Do not take sorted_values[idx]
        search(idx + 1, curr_sum, mask)
        
        # Option 2: Take sorted_values[idx]
        new_sum = curr_sum + sorted_values[idx]
        if new_sum < 1:
            search(idx + 1, new_sum, mask | (1 << idx))
            
    search(0, Fraction(0), 0)
    
    # Convert best_mask back to original indices
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(sorted_orig_indices[i])
            
    return sorted(result_indices)
