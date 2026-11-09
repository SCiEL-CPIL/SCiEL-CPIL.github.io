"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _simulate_greedy(capacity, weights, values, order_indices, exclude_idx):
    """
    Simulates a greedy knapsack solution given a specific ordering of items.
    Returns the total value.
    """
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check if any item fits
    if np.any(remaining_weights <= remaining_capacity):
        pass
    else:
        return -1

    # Compute density
    with np.errstate(divide='ignore', invalid='ignore'):
        density = remaining_values / remaining_weights
    density = np.where(remaining_weights > 0, density, 0)
    
    # Sort by density descending
    order = np.argsort(-density)
    
    # If we have time, do a 1-step lookahead for the top candidates
    # We want to pick item i such that if we pick i, the subsequent greedy fill is maximized.
    # This is equivalent to maximizing: value[i] + greedy_fill(remaining_capacity - weight[i], excluding i)
    
    # To keep it fast, we only consider the top K candidates by density
    # K=50 was measured to improve score. Let's try a slightly larger K if time permits, 
    # but keep it bounded for safety.
    
    # Filter to only feasible items for the lookahead (weight <= capacity)
    # Actually, the greedy simulation handles infeasible items by skipping them, 
    # but for the *first* pick, we must pick a feasible item.
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Get top K feasible items by density
    feasible_indices = order[feasible_mask]
    
    # If we have very few items, just check all
    if len(feasible_indices) <= 50:
        candidates = feasible_indices
    else:
        candidates = feasible_indices[:50]
    
    best_score = -1
    best_idx = -1
    
    # Pre-compute the full greedy order for the simulation
    # The simulation uses the global density order, skipping the excluded item.
    
    for idx in candidates:
        if not _elapsed_ok():
            break
            
        w = remaining_weights[idx]
        v = remaining_values[idx]
        
        # Simulate: pick idx first, then fill rest with greedy
        # The greedy fill uses the same density order, but skips idx.
        # We can simulate this quickly.
        
        sim_cap = remaining_capacity - w
        sim_val = v
        
        # Iterate through the rest of the density order
        # Optimization: if sim_cap <= 0, break
        for next_idx in order:
            if next_idx == idx:
                continue
            if sim_cap <= 0:
                break
            nw = remaining_weights[next_idx]
            if nw <= sim_cap:
                sim_cap -= nw
                sim_val += remaining_values[next_idx]
        
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx
            
    if best_idx == -1:
        # Fallback to pure greedy if something went wrong or no candidates
        return int(order[0])
        
    return int(best_idx)
# EVOLVE-BLOCK-END
