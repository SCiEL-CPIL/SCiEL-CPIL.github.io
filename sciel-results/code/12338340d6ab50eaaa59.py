"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    prizes: (5, 100) array of prizes
    weights: (5, 100) array of weights
    Returns: (100,) array of attractiveness scores (larger = more attractive)
    
    Design: For each item, compute prize/weight ratio per constraint, then take
    a smoothed minimum (soft-min) across constraints. This rewards items that
    are efficient in their most binding constraint.
    """
    safe_weights = np.maximum(weights, 1e-9)
    ratios = prizes / safe_weights  # (5, 100)
    
    # Soft-min with moderate temperature for smoothness
    tau = 1.0
    neg = -ratios / tau
    max_neg = np.max(neg, axis=0, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(neg - max_neg), axis=0)) + max_neg.squeeze(axis=0)
    soft_min = -log_sum_exp * tau
    
    result = np.maximum(soft_min, 1e-9)
    result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
