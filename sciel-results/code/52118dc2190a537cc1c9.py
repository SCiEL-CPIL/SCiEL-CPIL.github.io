"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from math import lcm, gcd

    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit

    nodes = list(range(2, 257))
    n = len(nodes)
    weights = [1.0 / x for x in nodes]
    
    # Precompute conflict/compatibility bitsets
    # compat_mask[i] is a bitset of nodes j such that lcm(nodes[i], nodes[j]) > 256
    # We only care about j > i for the graph structure, but for B&B we need full masks
    # relative to the candidate set.
    
    # 1. Build the graph adjacency as bitsets
    # Index mapping: node value v -> index v-2
    # Let's use the index in the 'nodes' list.
    # nodes[k] = k+2
    
    # compat[i] will hold a bitmask of all j such that nodes[i] and nodes[j] are compatible
    compat = [0] * n
    
    for i in range(n):
        ai = nodes[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            aj = nodes[j]
            if lcm(ai, aj) > 256:
                mask |= (1 << j)
        compat[i] = mask

    # 2. Maximum Weight Clique using Branch and Bound
    # We represent the candidate set as a bitmask.
    # best_clique is a bitmask of the best clique found so far.
    # best_weight is the sum of weights of the best clique.
    
    best_weight = 0.0
    best_mask = 0
    
    # Greedy initialization to get a good lower bound
    # Sort indices by weight descending
    order = sorted(range(n), key=lambda i: -weights[i])
    
    curr_mask = 0
    curr_weight = 0.0
    for idx in order:
        # Check if idx is compatible with all in curr_mask
        # (curr_mask & ~compat[idx]) == 0 means all nodes in curr_mask are in compat[idx]
        if (curr_mask & ~compat[idx]) == 0:
            curr_mask |= (1 << idx)
            curr_weight += weights[idx]
            
    best_weight = curr_weight
    best_mask = curr_mask

    # B&B Search
    # We use a recursive function. To avoid recursion depth issues and overhead,
    # we can use an iterative approach or optimize the recursion.
    # Given n=255, recursion depth is at most 255, which is fine for Python if optimized,
    # but function call overhead can be high.
    # However, the pruning is usually very effective.
    
    def max_weight_clique(candidates_mask, current_mask, current_weight):
        nonlocal best_weight, best_mask
        nonlocal deadline, start_time
        
        if time.time() > deadline:
            return

        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_mask = current_mask
            return

        # Upper bound: current_weight + sum of weights of all candidates
        # We can precompute weight sums for bitmasks? No, too large.
        # Calculate sum of weights for candidates in candidates_mask.
        # To speed this up, we can iterate through set bits.
        
        # Optimization: If the sum of weights of remaining candidates cannot beat best_weight, prune.
        # Let's calculate the potential max weight.
        # A tighter bound is better.
        # Simple bound: sum(weights[i] for i in candidates)
        
        # To avoid recalculating sum every time, we could pass it, but it changes as we branch.
        # Let's just calculate it.
        # Bit iteration
        temp_mask = candidates_mask
        upper_bound_add = 0.0
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            upper_bound_add += weights[idx]
            temp_mask ^= lsb
            
        if current_weight + upper_bound_add <= best_weight:
            return

        # Select a pivot node to branch on.
        # Heuristic: Choose the node with the highest weight among candidates?
        # Or the one that appears in the most edges?
        # Standard Max Clique uses coloring. For Max Weight, we can sort by weight.
        
        # Let's pick the candidate with the highest weight.
        # Find max weight index in candidates_mask
        max_w = -1
        max_idx = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                max_idx = idx
            temp_mask ^= lsb
            
        if max_idx == -1:
            return

        # Branch 1: Include max_idx
        # New candidates are candidates intersected with compat[max_idx]
        new_candidates = candidates_mask & compat[max_idx]
        new_current_mask = current_mask | (1 << max_idx)
        new_current_weight = current_weight + weights[max_idx]
        
        max_weight_clique(new_candidates, new_current_mask, new_current_weight)
        
        # Branch 2: Exclude max_idx
        # New candidates are candidates without max_idx
        new_candidates_excl = candidates_mask & ~(1 << max_idx)
        
        max_weight_clique(new_candidates_excl, current_mask, current_weight)

    # Start the search
    # Initial candidates: all nodes
    initial_candidates = (1 << n) - 1
    max_weight_clique(initial_candidates, 0, 0.0)

    # Extract result from best_mask
    result = []
    temp_mask = best_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        result.append(nodes[idx])
        temp_mask ^= lsb

    if not result:
        # Fallback if something went wrong (should not happen)
        result = [256]
        
    return result

# EVOLVE-BLOCK-END
