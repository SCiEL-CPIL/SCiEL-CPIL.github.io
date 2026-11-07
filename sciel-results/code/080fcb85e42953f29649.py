import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) or (100,), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
        
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Handle prizes shape
    if prizes.ndim > 1:
        item_prizes = prizes.sum(axis=0)  # Sum across constraint dimensions
    else:
        item_prizes = prizes.astype(float)
    
    # Ensure weights is 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Compute total weight per constraint
    total_weights = weights.sum(axis=1)  # shape: (5,)
    
    # Shadow price: inverse of total weight (higher when constraint is tighter)
    # If total weight is close to capacity, the constraint is more binding
    # Use a smoothed version to avoid division by zero
    epsilon = 1e-9
    total_weights_safe = total_weights + epsilon
    
    # Shadow prices: 1 / (total_weight + epsilon)
    # This gives higher weight to constraints that are more tightly packed
    shadow_prices = 1.0 / total_weights_safe  # shape: (5,)
    
    # Compute weighted resource cost for each item
    # cost_j = sum_i (shadow_price_i * weight_ij)
    weighted_costs = weights @ shadow_prices  # shape: (100,)
    
    # Lagrangian surplus: prize - weighted_cost
    # This directly penalizes items that consume scarce resources disproportionately
    surplus = item_prizes - weighted_costs  # shape: (100,)
    
    # Normalize to have mean 1 and ensure positivity
    mean_surplus = np.mean(surplus)
    std_surplus = np.std(surplus)
    
    if std_surplus > 1e-12:
        # Standardize and shift to be positive
        z_scores = (surplus - mean_surplus) / std_surplus
        # Shift so minimum is 0, then scale
        z_shifted = z_scores - np.min(z_scores)
        # Scale so mean is 1
        mean_z_shifted = np.mean(z_shifted)
        if mean_z_shifted > 1e-12:
            heuristic = z_shifted / mean_z_shifted
        else:
            heuristic = np.ones(n_items)
    else:
        heuristic = np.ones(n_items)
    
    # Ensure all values are positive (ACO requires positive heuristics)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Optional: Apply a power to sharpen the selection
    # Higher power = more selective, lower = more diverse
    # Let's try power 1.5 for moderate sharpening
    max_h = np.max(heuristic)
    if max_h > 1e-12:
        heuristic = (heuristic / max_h) ** 1.5
        # Re-normalize to mean 1
        mean_h = np.mean(heuristic)
        if mean_h > 1e-12:
            heuristic = heuristic / mean_h
    
    # Final safety: ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
