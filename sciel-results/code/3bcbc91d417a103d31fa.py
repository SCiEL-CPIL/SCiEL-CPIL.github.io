import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    N = 511
    weights = [1.0 / (512 - i) for i in range(1, N + 1)]
    
    # Sort indices by weight descending
    # indices are 1..511
    sorted_indices = sorted(range(1, N + 1), key=lambda x: weights[x-1], reverse=True)
    
    # Map sorted position to original value
    # pos 0 is largest weight
    vals = sorted_indices
    
    # Precompute conflict masks for each position in sorted order
    # conflict_mask[pos] is a bitmask of positions j such that vals[pos] and vals[j] are not coprime
    M = len(vals)
    conflict_mask = [0] * M
    
    for i in range(M):
        v_i = vals[i]
        mask = 0
        for j in range(M):
            if i == j:
                continue
            v_j = vals[j]
            if gcd(v_i, v_j) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    best_sum = 0.0
    best_set = []
    
    # Helper to update best
    def update_best(current_sum, selected_indices):
        nonlocal best_sum, best_set
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(selected_indices)
            
    # DFS
    # idx: current index in sorted list to consider
    # current_sum: sum of weights of selected vertices
    # selected: list of original values selected
    # available_mask: bitmask of indices >= idx that are compatible with selected vertices
    
    # Initial mask: all bits 0..M-1 set
    full_mask = (1 << M) - 1
    
    def dfs(idx, current_sum, selected, available_mask):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Upper bound: current_sum + sum of weights of all available vertices
        # We can optimize this bound calculation.
        # For now, just iterate bits.
        bound = current_sum
        m = available_mask
        while m:
            lsb = m & -m
            pos = lsb.bit_length() - 1
            bound += weights[vals[pos] - 1]
            m ^= lsb
            
        if bound <= best_sum:
            return
            
        # Base case
        if idx >= M:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(selected)
            return
            
        # Try including vals[idx]
        # Is it available?
        if (available_mask & (1 << idx)) != 0:
            # Include
            new_selected = selected + [vals[idx]]
            # New available mask: remove idx and all conflicts of idx
            # conflicts are in conflict_mask[idx]
            # We also need to ensure we only consider indices > idx in future steps?
            # The standard approach:
            # The available_mask passed to next step should only contain bits for indices > idx.
            # So we mask with ((1 << M) - 1) ^ ((1 << (idx + 1)) - 1)
            # Which is effectively removing bits 0..idx.
            
            # Let's refine the state transition.
            # Current available_mask contains bits for all indices that are compatible.
            # We are at index idx.
            
            # Branch 1: Include idx
            # New mask for next step (idx + 1):
            # Take available_mask, remove bit idx, remove all bits in conflict_mask[idx].
            # Then mask out bits <= idx to ensure we only look forward.
            forward_mask = ((1 << M) - 1) ^ ((1 << (idx + 1)) - 1)
            
            next_mask_include = (available_mask & ~(1 << idx) & ~conflict_mask[idx]) & forward_mask
            dfs(idx + 1, current_sum + weights[vals[idx] - 1], new_selected, next_mask_include)
            
            # Branch 2: Exclude idx
            # New mask for next step:
            # Take available_mask, remove bit idx.
            # Mask out bits <= idx.
            next_mask_exclude = (available_mask & ~(1 << idx)) & forward_mask
            dfs(idx + 1, current_sum, selected, next_mask_exclude)
            
        else:
            # If idx is not available (conflicted with previous selection), we must exclude it
            forward_mask = ((1 << M) - 1) ^ ((1 << (idx + 1)) - 1)
            next_mask_exclude = (available_mask & ~(1 << idx)) & forward_mask
            dfs(idx + 1, current_sum, selected, next_mask_exclude)

    # Start DFS
    # Initial selected is empty, available_mask is full_mask
    # But we must start with idx=0.
    # The available_mask should contain all bits initially.
    dfs(0, 0.0, [], full_mask)
    
    # Return the set of integers
    return best_set
