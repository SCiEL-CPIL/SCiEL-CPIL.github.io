import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    n = len(weights)
    
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items only
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_count = len(feasible_indices)
    
    if f_count == 1:
        return int(feasible_indices[0])
    
    # Work with only feasible items for efficiency
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    
    # Beam search: maintain B partial solutions
    # Each beam: (total_value, total_weight, chosen_mask)
    # We track chosen items as a boolean array per beam
    
    B = 64  # beam width
    t0 = time.time()
    deadline = 150.0 - 2.0  # safety margin
    
    # Initialize beams: each beam starts by picking one item
    # Beam state: arrays of shape (B,) for value, weight, and (B, f_count) for chosen
    beam_values = np.zeros(B, dtype=np.float64)
    beam_weights = np.zeros(B, dtype=np.float64)
    beam_chosen = np.zeros((B, f_count), dtype=bool)
    
    # Initialize: first B items (or all if f_count < B) as separate beams
    init_count = min(B, f_count)
    for i in range(init_count):
        beam_values[i] = f_values[i]
        beam_weights[i] = f_weights[i]
        beam_chosen[i, i] = True
    
    active_beams = init_count
    
    # If f_count <= B, we've already tried all single items
    # Now do beam search steps
    max_steps = min(f_count - 1, 50)  # limit steps to keep time bounded
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        
        # For each beam, compute which items can still be added
        # remaining_cap for beam b = remaining_capacity - beam_weights[b]
        remaining_caps = remaining_capacity - beam_weights  # shape (B,)
        
        # An item j can be added to beam b if:
        #   1. beam_chosen[b, j] == False
        #   2. f_weights[j] <= remaining_caps[b]
        
        # Vectorized: (B, f_count) matrix
        weight_ok = f_weights[None, :] <= remaining_caps[:, None]  # (B, f_count)
        not_chosen = ~beam_chosen  # (B, f_count)
        can_add = weight_ok & not_chosen  # (B, f_count)
        
        # Check if any beam can add anything
        if not np.any(can_add):
            break
        
        # For each (beam, item) pair that's valid, compute the new state value
        # new_value[b, j] = beam_values[b] + f_values[j] if can_add[b, j] else -inf
        candidate_values = beam_values[:, None] + f_values[None, :]  # (B, f_count)
        candidate_values[~can_add] = -np.inf
        
        # Also track new weight
        candidate_weights = beam_weights[:, None] + f_weights[None, :]  # (B, f_count)
        
        # Select top B candidates across all beams
        # Flatten and pick top B
        flat_vals = candidate_values.ravel()
        flat_weights = candidate_weights.ravel()
        flat_indices = np.arange(B * f_count)
        
        # Get top B indices
        top_k = min(B, B * f_count)
        # Use argpartition for speed, then sort top B
        if top_k < flat_vals.size:
            part_idx = np.argpartition(-flat_vals, top_k - 1)[:top_k]
            part_vals = flat_vals[part_idx]
            sort_order = np.argsort(-part_vals)
            top_idx = part_idx[sort_order]
        else:
            top_idx = np.argsort(-flat_vals)[:top_k]
        
        # Build new beams
        new_beam_values = np.zeros(B, dtype=np.float64)
        new_beam_weights = np.zeros(B, dtype=np.float64)
        new_beam_chosen = np.zeros((B, f_count), dtype=bool)
        
        valid_count = 0
        for k in range(top_k):
            idx = top_idx[k]
            if flat_vals[idx] == -np.inf:
                break
            b = idx // f_count
            j = idx % f_count
            
            new_beam_values[valid_count] = flat_vals[idx]
            new_beam_weights[valid_count] = flat_weights[idx]
            new_beam_chosen[valid_count] = beam_chosen[b].copy()
            new_beam_chosen[valid_count, j] = True
            valid_count += 1
        
        if valid_count == 0:
            break
        
        beam_values = new_beam_values[:valid_count]
        beam_weights = new_beam_weights[:valid_count]
        beam_chosen = new_beam_chosen[:valid_count]
        active_beams = valid_count
        
        # Early stop if no improvement possible
        if active_beams == 0:
            break
    
    # Find the best beam: highest total value
    # But we need to map back to the ORIGINAL item index
    # The best beam's first chosen item is what we return
    # Actually, we need to return the FIRST item in the optimal sequence
    # 
    # Wait - the problem asks us to pick the NEXT item to add.
    # The evaluator calls select_next_item repeatedly, removing chosen items.
    # So we just need to return the best first pick.
    #
    # Actually, let me re-read: "picks the next item to add"
    # The evaluator builds a solution by repeatedly calling this function.
    # Each call, the remaining arrays shrink.
    # So we just need to pick the single best next item.
    #
    # The beam search above simulates the FULL future to evaluate which first pick is best.
    # The first item of the best beam is our answer.
    
    # Find best beam by total value
    best_beam = int(np.argmax(beam_values))
    
    # The first item chosen in that beam is our answer
    # Find the first True in beam_chosen[best_beam]
    first_chosen = np.where(beam_chosen[best_beam])[0]
    if len(first_chosen) == 0:
        # Fallback: pick highest density feasible item
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(f_weights > 0, f_values / f_weights, 0)
        best_f = int(np.argmax(densities))
        return int(feasible_indices[best_f])
    
    first_pick = int(first_chosen[0])
    return int(feasible_indices[first_pick])
