import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Beam Search Parameters
    # Since capacity is small (12.5) and items are 100, the number of items
    # that can actually fit is small.
    # We use a beam width B to keep track of the best partial solutions.
    B = 50  # Beam width: number of partial solutions to keep
    K_EXPAND = 10  # Number of candidates to expand from each beam state
    
    # State: (current_value, current_capacity, used_indices_mask)
    # Using a mask is expensive for copying. 
    # Alternative: Since we only need to return the NEXT item, 
    # we can run a beam search that tracks the set of used items.
    # However, copying a set of 100 items 50 times per step is O(50*100) = 5000 ops.
    # 100 steps * 5000 = 500k ops. This is very fast.
    
    # Actually, we can be smarter. We don't need to track the full mask if we
    # just want the best *first* item. But beam search requires knowing what's used.
    
    # Let's use a list of used indices for each beam state.
    # State: (value, capacity, used_indices_list)
    
    # Initialize beam with one state: empty set
    # We sort items by density initially to help pruning? 
    # No, let's just expand top K density items from each state.
    
    # Pre-compute densities for quick sorting
    densities = remaining_values / remaining_weights
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    # Beam: list of tuples (value, capacity, used_indices_set)
    # Using a set for O(1) lookup of used items
    beam = [(0.0, remaining_capacity, frozenset())]
    
    best_next_item = -1
    best_final_value = -1.0
    
    # We need to simulate the entire knapsack filling process?
    # The task asks for `select_next_item`. The evaluator builds the solution step-by-step.
    # So `select_next_item` is called repeatedly.
    # Wait, the interface is: `select_next_item(remaining_capacity, remaining_weights, remaining_values)`.
    # The `remaining_*` arrays change as items are picked.
    # This means the function is stateless with respect to previous picks, 
    # EXCEPT that the input arrays reflect the current state.
    # 
    # CRITICAL INSIGHT: 
    # If the function is called repeatedly with updated arrays, 
    # a beam search *inside* this function would re-simulate the future 
    # from the current state.
    # 
    # However, if I run a beam search to the end of the knapsack 
    # (until no more items fit), I can determine which *first* item 
    # leads to the best *total* value from this point forward.
    # 
    # This is a "lookahead to the end" strategy.
    # Since capacity is only 12.5, the depth of the search is limited 
    # by the minimum weight of items. If min weight is ~0.1, depth is ~125.
    # If min weight is ~1.0, depth is ~12.
    # Beam search with B=50 and K=10 at depth 12 is 50*10 = 500 nodes per level.
    # 12 levels * 500 = 6000 nodes. Very fast.
    # Even if depth is 100, 100 * 500 = 50,000 nodes. Still fast.
    
    # Let's implement a beam search that simulates the rest of the packing.
    # For each candidate first item, we run a beam search to maximize total value.
    # Actually, we can just run ONE beam search where the "root" is the current state,
    # and we expand all possibilities. But we need to pick the *next* item.
    # 
    # Strategy:
    # 1. Generate initial candidates: Top K items by density that fit.
    # 2. For each candidate, simulate the rest of the packing using a greedy or beam heuristic.
    #    But doing a full beam search for each candidate is redundant.
    #    
    # Better Strategy:
    # Run a beam search from the current state.
    # The beam contains states (value, capacity, used_set).
    # We expand each state by adding the top K fitting items.
    # We keep the top B states by value.
    # We do this until no items fit or depth limit.
    # 
    # But how do we pick the *next* item?
    # The next item is the one that appears in the *best* complete solution path.
    # However, beam search keeps multiple paths. 
    # We can track the "parent" of each state.
    # At the end, we look at the best final state and backtrack to see which item was added first.
    
    # Let's try this.
    
    # State: (value, capacity, used_frozenset, parent_state_id, item_added)
    # We need to store the history to backtrack.
    
    # Since n=100, used_frozenset is small.
    # Let's limit the depth to avoid explosion if weights are tiny.
    MAX_DEPTH = 50 # Capacity 12.5, if min weight 0.25, max 50 items.
    
    # Beam entries: [value, capacity, used_set, parent_idx, item_idx]
    # parent_idx is the index in the previous beam list (or -1 for root)
    # We need to store all previous beams to backtrack.
    
    # To save memory/time, we can just store the path of items for each state?
    # No, that's expensive.
    # Let's store `parent` as a tuple of (beam_id, state_idx).
    
    # Actually, simpler:
    # Since we only care about the FIRST item, we can just look at the children of the root.
    # The root has 0 items. Its children are the first items picked.
    # If we run the beam search, the states in `beam` after step 1 are the candidates for the first item.
    # But beam search prunes. 
    # If we set B large enough, we might keep all top candidates.
    # 
    # Alternative: 
    # Just run a beam search that tracks the full path.
    # Given the small size, let's just store the list of items in the state.
    
    beam = []
    # Root state
    beam.append({
        'value': 0.0,
        'cap': remaining_capacity,
        'used': frozenset(),
        'path': []
    })
    
    for depth in range(MAX_DEPTH):
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for state in beam:
            used = state['used']
            cap = state['cap']
            val = state['value']
            path = state['path']
            
            # Find top K items that fit and are not used
            # We can iterate through sorted_indices (by density)
            candidates = []
            for idx in sorted_indices:
                if idx in used:
                    continue
                w = remaining_weights[idx]
                if w <= cap:
                    candidates.append(idx)
                    if len(candidates) >= K_EXPAND:
                        break
            
            for idx in candidates:
                new_cap = cap - remaining_weights[idx]
                new_val = val + remaining_values[idx]
                new_used = used | {idx}
                new_path = path + [idx]
                
                next_beam.append({
                    'value': new_val,
                    'cap': new_cap,
                    'used': new_used,
                    'path': new_path
                })
                
        if not next_beam:
            break
            
        # Prune next_beam to top B by value
        # Sort by value descending
        next_beam.sort(key=lambda x: x['value'], reverse=True)
        beam = next_beam[:B]
        
        # Check deadline
        if time.time() > deadline:
            break

    # After the loop, `beam` contains the best partial/complete solutions.
    # We want to pick the next item.
    # The "next item" is the first item in the path of the best solution?
    # Wait. The function is called at the *current* state.
    # The `path` in the beam states represents items picked *after* the current state.
    # So, if the best state in `beam` has path `[i1, i2, i3]`, 
    # then the best next item is `i1`.
    
    # However, if the beam search didn't go deep enough, or if there are ties,
    # we should pick the item that appears most frequently in the top beam states?
    # Or just the first item of the single best state.
    
    if beam:
        best_state = beam[0]
        if best_state['path']:
            return int(best_state['path'][0])
    
    # Fallback: Highest density item that fits
    for idx in sorted_indices:
        if remaining_weights[idx] <= remaining_capacity:
            return int(idx)
            
    return -1
