"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_heuristics(prizes, weights, deadline=None):
    """
    Compute heuristic attractiveness scores using Pareto dominance filtering.
    """
    start_time = time.time()
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.ndim != 2 or weights.ndim != 2:
        raise ValueError("prizes and weights must be 2D arrays")
    
    if prizes.shape[0] != weights.shape[0]:
        raise ValueError("prizes and weights must have same number of constraints")
    
    if prizes.shape[1] != weights.shape[1]:
        raise ValueError("prizes and weights must have same number of items")
    
    num_constraints, n_items = prizes.shape
    
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("prizes and weights must contain only finite values")
    
    # 1. Calculate total prize per item
    total_prize = prizes.sum(axis=0) # shape (n_items,)
    
    # 2. Calculate dominance
    # Item i dominates j if:
    # prize_i >= prize_j
    # weight_i[c] <= weight_j[c] for all c
    # AND (prize_i > prize_j OR any weight_i[c] < weight_j[c])
    
    # For efficiency, we can vectorize the weight comparison partially,
    # but n=100 is small, so a double loop is fine and clear.
    dominator_count = np.zeros(n_items, dtype=np.int32)
    
    # Pre-compute weight matrix for easy access
    # weights is (C, N). We need weights[:, i] <= weights[:, j]
    
    for i in range(n_items):
        if time.time() > (start_time + 10): # Hard timeout for this part just in case
             break
        prize_i = total_prize[i]
        w_i = weights[:, i]
        
        # Vectorized comparison for all j
        # prize_i >= total_prize (all)
        # w_i <= weights (all constraints for all items)
        
        # Check prize condition
        prize_ok = total_prize <= prize_i
        
        # Check weight condition: w_i <= weights[:, j] for all c
        # weights is (C, N). w_i is (C,).
        # We want a boolean array of shape (N,) where True means w_i <= weights[:, j] for all c.
        # weights.T is (N, C).
        # (w_i <= weights.T) gives (N, C).
        # .all(axis=1) gives (N,).
        weight_ok = np.all(w_i <= weights.T, axis=1)
        
        # Combined: i dominates j if prize_ok[j] and weight_ok[j]
        # But we need strict inequality for at least one dimension to be a "proper" dominance
        # Usually in Pareto, if they are identical, they don't dominate each other.
        # If prize_i == prize_j and weights are identical, no dominance.
        
        # Let's find candidates where i is "at least as good" as j
        candidates = np.where(prize_ok & weight_ok)[0]
        
        for j in candidates:
            if i == j:
                continue
            # Check strict inequality
            if total_prize[i] > total_prize[j] or np.any(w_i < weights[:, j]):
                dominator_count[j] += 1
                
    # 3. Compute base heuristic
    # Use sum of weights as the cost metric
    sum_weights = weights.sum(axis=0)
    eps = 1e-9
    base_heuristic = total_prize / (sum_weights + eps)
    
    # 4. Apply Pareto filter
    # Items with 0 dominators are on the Pareto front.
    # We want to boost them.
    # We can multiply by a factor that decays with the number of dominators.
    # heuristic = base_heuristic * (1.0 / (1.0 + dominator_count))
    # This gives Pareto optimal items a factor of 1.0 (relative to base)
    # and dominated items a factor < 1.0.
    
    # To make the effect stronger, we can use:
    # factor = 1.0 / (1.0 + 5.0 * dominator_count)
    # This heavily penalizes items dominated by many others.
    
    dominance_factor = 1.0 / (1.0 + 2.0 * dominator_count)
    
    heuristic = base_heuristic * dominance_factor
    
    # Ensure positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in ant colony optimization
    of the multiple-constraint knapsack problem.
    """
    start_time = time.time()
    
    try:
        result = _compute_heuristics(prizes, weights, start_time + 290)
        
        result = np.asarray(result, dtype=np.float64)
        
        if result.ndim != 1:
            raise ValueError("Heuristic must be a 1D array")
        
        if not np.all(np.isfinite(result)):
            raise ValueError("Heuristic must contain only finite values")
        
        prizes_arr = np.asarray(prizes)
        expected_n_items = prizes_arr.shape[1] if prizes_arr.ndim == 2 else len(prizes_arr)
        
        if result.shape[0] != expected_n_items:
            raise ValueError(f"Heuristic shape {result.shape} doesn't match expected ({expected_n_items},)")
        
        return result
        
    except Exception:
        prizes_arr = np.asarray(prizes, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        
        if prizes_arr.ndim == 2:
            n_items = prizes_arr.shape[1]
        else:
            n_items = len(prizes_arr)
            
        return np.ones(n_items, dtype=np.float64)
# EVOLVE-BLOCK-END
