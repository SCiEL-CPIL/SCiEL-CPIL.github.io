import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # All candidates 2..256
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute conflicts: conflict[i][j] = True if lcm(candidates[i], candidates[j]) <= 256
    # Use bitmasks for efficiency
    # n = 255, so bitmask fits in Python int
    
    # Build conflict bitmasks
    conflict_mask = [0] * n
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            g = gcd(ai, aj)
            lcm_val = (ai // g) * aj
            if lcm_val <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Weight for each node: 1/value
    weights = [1.0 / candidates[i] for i in range(n)]
    
    # Sort by weight descending (value ascending) for greedy
    # We want to pick high weight (small number) first
    order = sorted(range(n), key=lambda i: candidates[i])  # ascending value = descending weight
    
    # Greedy Construction
    selected_mask = 0
    available_mask = (1 << n) - 1
    selected_indices = []
    
    for i in order:
        if available_mask & (1 << i):
            # Include i
            selected_mask |= (1 << i)
            selected_indices.append(i)
            # Remove i and its conflicts from available
            available_mask &= ~(1 << i)
            available_mask &= ~conflict_mask[i]
            
    # Local Search: 1-Swap
    # Try to swap a selected node s with an unselected node u
    # if u is compatible with all selected nodes except s, and weight[u] > weight[s]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # List of selected indices
        curr_selected = [i for i in range(n) if selected_mask & (1 << i)]
        # List of unselected indices
        curr_unselected = [i for i in range(n) if not (selected_mask & (1 << i))]
        
        # For each selected node, try to find a better unselected node to swap with
        # We iterate through unselected nodes first to find best candidate for each s?
        # Or just check all pairs.
        # To allow multiple swaps in one pass (if they don't interfere), 
        # we can mark nodes as "swapped" temporarily.
        # However, simple sequential swaps are safer and often sufficient.
        
        # Optimization: Pre-compute for each unselected node u, which selected nodes it conflicts with.
        # If u conflicts with only one selected node s, it's a candidate for swap with s.
        
        potential_swaps = []
        
        for u in curr_unselected:
            # Find selected nodes that conflict with u
            conflicts_with_selected = conflict_mask[u] & selected_mask
            # Count how many selected nodes u conflicts with
            # If 0, we could just add u (but greedy should have handled this, unless logic changed)
            # If 1, let s be that node. We can swap s with u if w(u) > w(s).
            # If >1, we cannot swap u in with a single swap.
            
            if conflicts_with_selected == 0:
                # This shouldn't happen in a maximal set, but if it does, we can add u.
                # Adding u increases sum.
                # Let's just add it.
                selected_mask |= (1 << u)
                available_mask &= ~(1 << u)
                available_mask &= ~conflict_mask[u]
                selected_indices.append(u)
                improved = True
                # Restart loop as masks changed
                break
            elif conflicts_with_selected != 0:
                # Check if it's a single bit
                # (x & (x-1)) == 0 checks for power of 2
                if (conflicts_with_selected & (conflicts_with_selected - 1)) == 0:
                    s = conflicts_with_selected.bit_length() - 1
                    if weights[u] > weights[s]:
                        potential_swaps.append((s, u, weights[u] - weights[s]))
        
        if improved:
            continue
            
        # Sort swaps by gain descending to apply best swaps first
        potential_swaps.sort(key=lambda x: -x[2])
        
        for s, u, gain in potential_swaps:
            # Check if s is still selected and u is still unselected
            if not (selected_mask & (1 << s)) or (selected_mask & (1 << u)):
                continue
                
            # Apply swap
            selected_mask &= ~(1 << s)
            selected_mask |= (1 << u)
            # Update available mask
            # Remove s from available? No, s is now unselected.
            # But we need to update available_mask to reflect new selected_mask.
            # available_mask should be nodes compatible with selected_mask.
            # Recalculating available_mask from scratch is O(N) or O(1) with bit tricks?
            # available_mask = all_nodes & ~union(conflict_mask[i] for i in selected)
            # This is expensive.
            # Instead, we can just re-run the greedy/local search from scratch if we do many swaps?
            # Or maintain available_mask carefully.
            
            # Let's just recompute available_mask if we do a swap.
            # But recomputing is O(N^2/word_size) or similar.
            # Given N=255, it's fast.
            
            # Actually, we can update incrementally:
            # Removing s: we add back s and its conflicts? No, s was selected, so its conflicts were removed from available.
            # Now s is unselected. Its conflicts might become available if they don't conflict with other selected.
            # Adding u: we remove u and its conflicts from available.
            
            # This incremental update is error-prone.
            # Let's just recompute available_mask after each swap.
            # available_mask = (1 << n) - 1
            # for i in range(n):
            #    if selected_mask & (1 << i):
            #       available_mask &= ~conflict_mask[i]
            #       available_mask &= ~(1 << i)
            
            # This is O(N) per swap. With ~100 swaps, it's 100*255 = 25500 ops. Trivial.
            
            # Perform swap
            improved = True
            
            # Recompute available_mask
            available_mask = (1 << n) - 1
            for i in range(n):
                if selected_mask & (1 << i):
                    available_mask &= ~conflict_mask[i]
                    available_mask &= ~(1 << i)
            
            # Break to restart the while loop with updated masks
            break
            
    # Convert back to actual values
    result = [candidates[i] for i in range(n) if selected_mask & (1 << i)]
    
    if not result:
        return [256]
        
    return result
