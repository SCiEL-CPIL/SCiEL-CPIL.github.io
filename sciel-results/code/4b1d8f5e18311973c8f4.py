"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_fill(remaining_capacity, remaining_weights, remaining_values):
    """
    Greedily fill the knapsack by value density, skipping items that don't fit.
    Returns the total value achievable by this greedy strategy.
    """
    if remaining_capacity <= 0 or len(remaining_weights) == 0:
        return 0.0
    
    # Compute densities and sort by density descending
    # Use stable sort for ties
    densities = remaining_values / remaining_weights
    order = np.argsort(-densities, kind='stable')
    
    total_value = 0.0
    cap = remaining_capacity
    for idx in order:
        w = remaining_weights[idx]
        v = remaining_values[idx]
        if w <= cap:
            total_value += v
            cap -= w
        if cap <= 0:
            break
    
    return total_value


def _one_step_lookahead(remaining_capacity, remaining_weights, remaining_values, time_budget=0.8):
    """
    For each feasible item, simulate picking it first, then greedily filling the rest.
    Return the index of the item that leads to the highest total value.
    If time budget is exceeded, fall back to pure density.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    # If only one feasible item, return it
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # If too many items, fall back to density to save time
    # The look-ahead is O(n^2 log n) in worst case if we do full greedy each time
    # But we can be smarter: for each candidate, we just need to simulate greedy on the rest
    # Let's set a threshold: if n_feasible > 50, use a hybrid approach
    # Actually, let's just do it but with a time check
    
    start_time = time.time()
    
    # Precompute densities
    densities = remaining_values / remaining_weights
    
    # For each feasible candidate, compute the value if we pick it first
    best_score = -1.0
    best_idx = -1
    
    # To speed up, we can sort feasible items by density and use that order
    # For candidate i, the greedy fill will consider all other feasible items in density order
    # We can precompute the sorted order of all items by density
    all_order = np.argsort(-densities, kind='stable')
    
    # Create a mapping from original index to position in sorted order
    # This helps us quickly iterate through items in density order excluding candidate i
    
    for cand_pos in range(n_feasible):
        if time.time() - start_time > time_budget:
            # Fall back to pure density
            break
        
        cand_idx = feasible_indices[cand_pos]
        cand_w = remaining_weights[cand_idx]
        cand_v = remaining_values[cand_idx]
        
        # Simulate: pick cand_idx, then greedily fill remaining capacity
        cap_after = remaining_capacity - cand_w
        total_v = cand_v
        
        # Now greedily fill with remaining items in density order
        # We iterate through all_order, skipping cand_idx and infeasible items
        for item_idx in all_order:
            if item_idx == cand_idx:
                continue
            if remaining_weights[item_idx] > remaining_capacity:
                # This item was never feasible, skip
                continue
            if remaining_weights[item_idx] <= cap_after:
                total_v += remaining_values[item_idx]
                cap_after -= remaining_weights[item_idx]
            if cap_after <= 0:
                break
        
        if total_v > best_score:
            best_score = total_v
            best_idx = int(cand_idx)
    
    if best_idx != -1:
        return best_idx
    
    # Fallback: pure density
    return int(feasible_indices[np.argmax(densities[feasible_indices])])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    """
    if len(remaining_weights) == 0 or remaining_capacity <= 0:
        return -1
    
    # Check if any item fits
    if np.max(remaining_weights) > remaining_capacity:
        # Some items may still fit
        feasible_mask = remaining_weights <= remaining_capacity
        if not np.any(feasible_mask):
            return -1
    else:
        feasible_mask = np.ones(len(remaining_weights), dtype=bool)
    
    # Use one-step look-ahead with time budget
    # The time budget should be conservative since this function is called many times
    # For 100 items, the look-ahead is O(n^2) which is ~10000 operations, very fast
    # So we can afford a reasonable time budget per call
    return _one_step_lookahead(remaining_capacity, remaining_weights, remaining_values, time_budget=0.5)
# EVOLVE-BLOCK-END
