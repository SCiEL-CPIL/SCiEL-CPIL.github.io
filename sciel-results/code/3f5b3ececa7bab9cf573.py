"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    
    # Denominators are 2 to 48
    # Indices are 0 to 47 corresponding to denominators 2 to 48
    # index i corresponds to denominator d = i + 2
    
    n = 47
    denominators = list(range(2, 49))
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # We want to search in descending order of denominators: 48, 47, ..., 2
    # This corresponds to indices: 46, 45, ..., 0
    # Let's create the order of indices to visit
    order = list(range(n - 1, -1, -1))
    
    # Precompute suffix sums of reciprocals in this specific order
    # suffix_sum[i] is the sum of reciprocals for order[i], order[i+1], ...
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        idx = order[i]
        suffix_sums[i] = suffix_sums[i + 1] + reciprocals[idx]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Greedy heuristic to find a good initial solution
    # "Largest that fits"
    remaining = Fraction(1)
    available = set(range(n))
    greedy_indices = []
    greedy_sum = Fraction(0)
    
    # To optimize greedy, we can just iterate sorted by value descending (denominator ascending)
    # and pick if it fits. This is O(N^2) which is fine for N=47.
    # Actually, standard "First Fit Decreasing" sorts items by size descending.
    # Here items are 1/d. Sizes are 1/2, 1/3, ...
    # So we consider d=2, then d=3, etc.
    for d in range(2, 49):
        idx = d - 2
        val = Fraction(1, d)
        if greedy_sum + val < 1:
            greedy_sum += val
            greedy_indices.append(idx)
            
    # This simple greedy might be weak. Let's try a slightly smarter one:
    # Always pick the largest fraction that fits in the remaining space.
    remaining = Fraction(1)
    available_indices = list(range(n))
    smart_greedy_indices = []
    smart_greedy_sum = Fraction(0)
    
    while available_indices:
        best_idx = -1
        best_val = Fraction(0)
        # Find largest 1/d that fits in 'remaining'
        # Largest 1/d means smallest d
        # But it must be <= remaining
        for idx in available_indices:
            val = reciprocals[idx]
            if val <= remaining:
                if best_idx == -1 or val > best_val:
                    best_val = val
                    best_idx = idx
        if best_idx == -1:
            break
        smart_greedy_sum += best_val
        smart_greedy_indices.append(best_idx)
        remaining -= best_val
        available_indices.remove(best_idx)
        
    if smart_greedy_sum > best_sum:
        best_sum = smart_greedy_sum
        best_indices = smart_greedy_indices.copy()
        
    # Also try the simple ascending greedy just in case
    if greedy_sum > best_sum:
        best_sum = greedy_sum
        best_indices = greedy_indices.copy()

    # DFS Branch and Bound
    # State: pos (index in 'order'), current_sum, current_indices
    # Pruning: if current_sum + suffix_sums[pos] <= best_sum, prune
    
    sys.setrecursionlimit(10000)
    
    def dfs(pos, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Pruning 1: Can we beat the best?
        # Max possible sum from here is current_sum + suffix_sums[pos]
        if current_sum + suffix_sums[pos] <= best_sum:
            return
            
        # Pruning 2: Is sum valid?
        if current_sum >= 1:
            return
            
        # Base case
        if pos == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices.copy()
            return
        
        idx = order[pos]
        val = reciprocals[idx]
        
        # Branch 1: Include idx
        # Check if adding val keeps sum < 1
        if current_sum + val < 1:
            current_indices.append(idx)
            dfs(pos + 1, current_sum + val, current_indices)
            current_indices.pop()
            
        # Branch 2: Exclude idx
        dfs(pos + 1, current_sum, current_indices)
        
    # Start DFS
    dfs(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
