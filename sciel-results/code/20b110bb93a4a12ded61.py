import numpy as np

def heuristics(prizes, weights):
    # Calculate ratios for each constraint
    eps = 1e-9
    ratios = prizes / (weights + eps)
    
    # Calculate average weight per constraint to determine "tightness"
    # Higher average weight implies the constraint is more likely to be a bottleneck
    avg_weights = np.mean(weights, axis=1)
    
    # Weights for the harmonic mean: constraints that are tighter (higher avg weight)
    # should have a stronger influence on the heuristic (higher weight in the sum)
    # Using inverse average weight as the weight for the constraint in the harmonic mean
    # Actually, if a constraint is tight, a low ratio is very bad.
    # Let's use the average weight itself as the weight in the weighted harmonic mean?
    # Weighted Harmonic Mean: H = (sum w_i) / (sum w_i / x_i)
    # If we want tight constraints to dominate, we want high w_k for tight constraints.
    # So let w_k = avg_weights[k].
    
    w = avg_weights + eps
    inv_ratios = 1.0 / ratios
    
    # Weighted sum of inverse ratios
    sum_weighted_inv = np.sum(w[:, np.newaxis] * inv_ratios, axis=0)
    sum_weights = np.sum(w)
    
    eta = sum_weights / sum_weighted_inv
    
    # Ensure positive finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
