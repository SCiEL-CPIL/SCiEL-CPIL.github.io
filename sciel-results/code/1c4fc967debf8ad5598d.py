import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the MCKP ACO solver.
    Uses a multi-constraint-aware heuristic that considers the effective
    resource consumption across all constraints, not just the maximum.
    """
    start_time = time.time()
    deadline = 325.0  # 300s limit + margin, but we are fast so this is just a guard
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Each constraint has capacity 25
    capacity = 25.0
    
    # For each item, compute the fraction of capacity consumed in each constraint
    # weights shape is (5, 100), so weights[c, i] is weight of item i in constraint c
    capacity_usage = weights / capacity  # shape (5, 100)
    
    # An item is infeasible if any single constraint usage > 1.0
    # For feasible items, the "effective cost" should reflect the tightest constraint
    # Use the maximum usage (bottleneck) but also penalize items that are heavy in multiple constraints
    
    max_usage = np.max(capacity_usage, axis=0)  # shape (100,)
    
    # Penalize items that are close to the limit in multiple constraints
    # Count how many constraints have usage > 0.8 (tight)
    tight_count = np.sum(capacity_usage > 0.8, axis=0)  # shape (100,)
    
    # Compute a composite cost: max_usage plus a small penalty for being tight in multiple constraints
    # This encourages diversity in which constraints are used
    composite_cost = max_usage + 0.1 * tight_count
    
    # Heuristic: prize / composite_cost
    heuristic = prizes / (composite_cost + 1e-9)
    
    # Ensure no non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Check time (should be negligible)
    if time.time() - start_time > deadline:
        # Fallback to a simple uniform heuristic if we are somehow slow (should not happen)
        return np.ones(100)
        
    # Normalize heuristic to have mean 1 to keep pheromone*heuristic scale consistent
    mean_heuristic = np.mean(heuristic)
    if mean_heuristic > 0:
        heuristic = heuristic / mean_heuristic
    
    return heuristic
