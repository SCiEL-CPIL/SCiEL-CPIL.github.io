"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: select the next item to add.
    
    Strategy: 1-step look-ahead heuristic. For each feasible item, estimate
    the total value if we take it now plus the best value achievable from
    remaining items with the reduced capacity. We approximate the remaining
    value using a greedy density-based fill on the remaining items.
    
    This accounts for the opportunity cost of taking an item: items that
    leave more capacity for high-density future items are preferred.
    """
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)
    
    # Check if any items are feasible
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    n = len(feasible_weights)
    
    # For each candidate item, estimate total value = item_value + estimated_remaining_value
    # Estimated remaining value: greedy fill of remaining capacity with other items by density
    
    # Precompute densities for all feasible items
    densities = feasible_values / feasible_weights
    
    best_total = -1.0
    best_local_idx = 0
    
    for i in range(n):
        # Take item i
        item_value = feasible_values[i]
        item_weight = feasible_weights[i]
        remaining_cap = capacity - item_weight
        
        if remaining_cap < 1e-12:
            total_est = item_value
        else:
            # Estimate remaining value: greedily pick highest density items (excluding i) that fit
            # Create a mask of items excluding i
            other_mask = np.ones(n, dtype=bool)
            other_mask[i] = False
            other_weights = feasible_weights[other_mask]
            other_values = feasible_values[other_mask]
            other_densities = densities[other_mask]
            
            # Sort by density descending
            order = np.argsort(-other_densities)
            other_weights_sorted = other_weights[order]
            other_values_sorted = other_values[order]
            
            # Greedily fill
            est_remaining = 0.0
            cap_left = remaining_cap
            for j in range(len(other_weights_sorted)):
                if other_weights_sorted[j] <= cap_left:
                    est_remaining += other_values_sorted[j]
                    cap_left -= other_weights_sorted[j]
                if cap_left < 1e-12:
                    break
            
            total_est = item_value + est_remaining
        
        if total_est > best_total:
            best_total = total_est
            best_local_idx = i
    
    best_global_idx = int(feasible_indices[best_local_idx])
    return best_global_idx
# EVOLVE-BLOCK-END
