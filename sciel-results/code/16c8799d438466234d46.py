"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit

def _greedy_completion(capacity, weights, values, excluded_idx):
    """
    Estimate the maximum value achievable by greedily filling 'capacity'
    with items from 'weights'/'values', excluding 'excluded_idx'.
    """
    if capacity <= 0:
        return 0.0
    
    # Create a mask of valid items (fit in capacity and not excluded)
    # Note: We only consider items that individually fit, as greedy won't pick non-fitting ones anyway
    n = len(weights)
    
    # Efficiently compute scores for all items, then filter
    # We want items where weights[i] <= capacity and i != excluded_idx
    # To avoid creating large temporary arrays repeatedly, we can just loop or use vectorized ops
    
    # Vectorized approach for the lookahead:
    # 1. Check which items fit in the *current* remaining capacity (after picking the candidate)
    # 2. Check which items are not the excluded one
    # 3. Sort by density
    
    # This is called N times per step. N=100.
    # Total ops: 100 * 100 * log(100) ~ 70k ops per step. 100 steps -> 7M ops. 
    # This is well within 110s in Python/NumPy.
    
    valid_mask = (weights <= capacity) & (np.arange(n) != excluded_idx)
    
    if not np.any(valid_mask):
        return 0.0
    
    valid_indices = np.where(valid_mask)[0]
    
    # Calculate densities for valid items
    w = weights[valid_indices]
    v = values[valid_indices]
    
    # Avoid division by zero (weights > 0 is assumed in knapsack, but safe check)
    densities = v / np.maximum(w, 1e-9)
    
    # Sort by density descending
    sorted_indices = np.argsort(-densities)
    
    total_val = 0.0
    remaining_cap = capacity
    
    for idx in sorted_indices:
        wi = w[idx]
        vi = v[idx]
        if wi <= remaining_cap:
            total_val += vi
            remaining_cap -= wi
        if remaining_cap <= 0:
            break
            
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item to add to the knapsack using a lookahead greedy strategy.
    Returns the index of the best item, or -1 if no item fits or time limit is reached.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    # Find all feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -1.0
    best_idx = -1
    
    # For each feasible item, estimate the total value if we pick it
    for idx in feasible_indices:
        w_i = remaining_weights[idx]
        v_i = remaining_values[idx]
        
        # Remaining capacity after picking idx
        new_cap = remaining_capacity - w_i
        
        # Estimate value of completing the knapsack with remaining items
        # We pass the original arrays and the excluded index
        completion_val = _greedy_completion(new_cap, remaining_weights, remaining_values, idx)
        
        total_est_val = v_i + completion_val
        
        if total_est_val > best_score:
            best_score = total_est_val
            best_idx = idx
            
    return int(best_idx)
# EVOLVE-BLOCK-END
