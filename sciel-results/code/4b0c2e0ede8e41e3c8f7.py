"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10 seconds margin for safety

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Check deadline
    if time.time() > _DEADLINE:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for vectorized operations
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)

    # Identify feasible items (weight <= remaining_capacity)
    # Also ensure weight > 0 to avoid division issues, though problem implies positive weights
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_indices]
    f_values = values[feasible_indices]
    
    if len(f_weights) == 0:
        return -1

    # If only one feasible item, pick it
    if len(f_weights) == 1:
        return int(feasible_indices[0])

    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = f_values / f_weights
    
    # Sort indices by density descending
    # We want the original indices in feasible_indices sorted by their densities
    sorted_order = np.argsort(-densities)
    sorted_indices = feasible_indices[sorted_order]
    sorted_weights = f_weights[sorted_order]
    sorted_values = f_values[sorted_order]
    sorted_densities = densities[sorted_order]

    # Find the marginal item to estimate lambda
    # The marginal item is the first one in the sorted list where cumulative weight >= remaining_capacity
    cum_weights = np.cumsum(sorted_weights)
    
    # Find the first index where cum_weights >= remaining_capacity
    # If all items fit, the marginal item is the last one? 
    # Actually, if all items fit, we are in a "loose" capacity scenario.
    # In that case, the optimal lambda is 0? Or the density of the last item?
    # If all items fit, we can take all of them. The "cost" of capacity is low.
    # If we set lambda to the min density, items with higher density have positive score.
    
    exceed_idx = np.searchsorted(cum_weights, remaining_capacity, side='left')
    
    if exceed_idx >= len(sorted_weights):
        # All items fit. The marginal density is the density of the last item (lowest density)
        # Or effectively 0 if we consider we can take all.
        # Using the lowest density as lambda ensures all items have non-negative score,
        # and the highest density items have the highest score.
        lambda_est = sorted_densities[-1]
    else:
        # The item at exceed_idx is the one that pushes us over the limit.
        # Its density is the marginal rate of return.
        lambda_est = sorted_densities[exceed_idx]

    # Calculate Lagrangian scores: v_i - lambda * w_i
    scores = f_values - lambda_est * f_weights
    
    # Find the item with the maximum score
    best_local_idx = np.argmax(scores)
    
    # Map back to original index
    best_original_idx = feasible_indices[best_local_idx]
    
    return int(best_original_idx)
# EVOLVE-BLOCK-END
