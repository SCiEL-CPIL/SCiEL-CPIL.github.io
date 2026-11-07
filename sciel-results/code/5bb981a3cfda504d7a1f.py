import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic values for Ant Colony Optimization on a Multiple-Constraint Knapsack Problem.
    Uses the Harmonic Mean of per-constraint efficiency ratios (prize/weight).
    The harmonic mean is sensitive to low efficiencies, effectively penalizing items 
    that are poor in any single constraint, which helps avoid bottlenecks.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are compatible (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    if prizes.shape[0] == 1 and weights.shape[0] > 1:
        prizes = np.repeat(prizes, weights.shape[0], axis=0)
    elif weights.shape[0] == 1 and prizes.shape[0] > 1:
        weights = np.repeat(weights, prizes.shape[0], axis=0)
        
    num_constraints, num_items = prizes.shape
    
    eps = 1e-9
    
    # Calculate efficiency for each constraint and item
    # safe_weights avoids division by zero
    safe_weights = np.where(weights < eps, eps, weights)
    ratios = prizes / safe_weights  # shape (num_constraints, num_items)
    
    # Ensure ratios are positive for harmonic mean calculation
    ratios = np.maximum(ratios, eps)
    
    # Harmonic Mean: J / sum(1/ratio)
    # This is a smooth surrogate for the minimum, but less extreme than min,
    # and more sensitive to low values than geometric mean.
    inv_ratios = 1.0 / ratios
    sum_inv_ratios = np.sum(inv_ratios, axis=0) # shape (num_items,)
    
    # Prevent division by zero in the harmonic mean calculation
    safe_sum_inv = np.where(sum_inv_ratios < eps, eps, sum_inv_ratios)
    
    harmonic_mean_eff = num_constraints / safe_sum_inv
    
    # Ensure output is positive and finite
    heuristic = np.maximum(harmonic_mean_eff, eps)
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    return heuristic
