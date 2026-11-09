"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in ant colony optimization
    of the multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (num_constraints, n_items) with values in [0,1]
        weights: array of shape (num_constraints, n_items) with values in [0,1]
    
    Returns:
        array of shape (n_items,) with one heuristic score per item (larger = more attractive)
    """
    # Convert to numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Basic validation
    if prizes.ndim != 2 or weights.ndim != 2:
        raise ValueError("Input arrays must be 2D")
    if prizes.shape != weights.shape:
        raise ValueError("Input arrays must have the same shape")
        
    num_constraints, n_items = prizes.shape
    
    # Check for non-finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Input arrays must contain only finite values")
        
    # Calculate total prize per item
    # Assumption: Total prize is the sum of prizes across constraints
    total_prize = prizes.sum(axis=0)
    
    # Calculate the bottleneck weight per item
    # Using max weight focuses on the most restrictive constraint
    max_weight = weights.max(axis=0)
    
    # Calculate raw heuristic: Prize / Bottleneck Weight
    # Add epsilon to avoid division by zero
    eps = 1e-9
    raw_heuristic = total_prize / (max_weight + eps)
    
    # Normalize the heuristics to be in the range (0, 1]
    # This helps balance the influence of pheromone (starts at 1) and heuristic
    max_raw = np.max(raw_heuristic)
    if max_raw > 0:
        heuristic = raw_heuristic / max_raw
    else:
        heuristic = np.ones(n_items)
        
    # Ensure all values are positive and finite
    # The task says "Heuristic entries that are not positive become 1e-9"
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
