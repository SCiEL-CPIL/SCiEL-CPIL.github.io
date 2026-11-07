"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100), prize values
        weights: array of shape (5, 100), weight values per constraint
    
    Returns:
        array of shape (100,) with attractiveness score per item
    """
    # Convert to numpy arrays for efficiency and safety
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Calculate the total prize value for each item.
    # The task implies "total prize" is the objective, so we sum the prize components.
    item_prizes = np.sum(prizes, axis=0)
    
    # Calculate the sum of weights for each item across all constraints.
    # This represents the total resource consumption.
    sum_weights = np.sum(weights, axis=0)
    
    # Calculate the geometric mean of weights for each item.
    # This represents the multiplicative average weight, sensitive to the "balance" of weights.
    # We add a small epsilon to avoid log(0).
    eps = 1e-9
    log_weights = np.log(weights + eps)
    geo_weights = np.exp(np.mean(log_weights, axis=0))
    
    # Calculate the heuristic ratios.
    # Ratio 1: Prize / Sum Weights
    # Ratio 2: Prize / Geometric Mean Weights
    # We add epsilon to denominators to prevent division by zero.
    h_sum = item_prizes / (sum_weights + eps)
    h_geo = item_prizes / (geo_weights + eps)
    
    # Combine the two heuristics using their geometric mean.
    # This balances the "total cost" signal with the "average multiplicative cost" signal.
    # The geometric mean is robust to the different scales of the two ratios.
    combined = np.sqrt(h_sum * h_geo)
    
    # Ensure all values are finite and positive.
    # The task states: "Heuristic entries that are not positive become 1e-9".
    # We also guard against NaN or Inf.
    if not np.all(np.isfinite(combined)):
        combined = np.where(np.isfinite(combined) & (combined > 0), combined, 1e-9)
    else:
        combined = np.where(combined > 0, combined, 1e-9)
        
    return combined
# EVOLVE-BLOCK-END
