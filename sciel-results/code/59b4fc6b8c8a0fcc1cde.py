"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    The heuristic should guide ants toward items that are efficient across ALL constraints,
    not just on average. We use a soft-min of per-constraint efficiency ratios to identify
    items that don't have a severe bottleneck in any single constraint.
    
    Args:
        prizes: array of shape (5, 100) with prize values per constraint and item
        weights: array of shape (5, 100) with weight values per constraint and item
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Calculate per-constraint efficiency ratios
    # Shape: (5, 100)
    # Add epsilon to weights to avoid division by zero
    eps = 1e-9
    ratios = prizes / (weights + eps)
    
    # Use a Soft-Min approximation to identify the bottleneck constraint efficiency
    # SoftMin(x) = -1/beta * log(sum(exp(-beta * x)))
    # This smooths the minimum of the ratios, penalizing items that have
    # very low efficiency in any single constraint.
    beta = 2.0
    
    # For numerical stability, subtract the max of (-beta * ratios) before exp
    # Let v = -beta * ratios
    v = -beta * ratios
    v_max = np.max(v, axis=0, keepdims=True)
    
    # sum(exp(v - v_max))
    exp_v = np.exp(v - v_max)
    sum_exp = np.sum(exp_v, axis=0)
    
    # soft_min = -1/beta * (v_max + log(sum_exp))
    # Note: v_max is shape (1, 100), sum_exp is shape (100,)
    soft_min = -1.0 / beta * (v_max[0] + np.log(sum_exp))
    
    # Ensure all values are positive and finite
    # Clip to avoid extreme values
    soft_min = np.clip(soft_min, 1e-9, 1e6)
    
    # Replace any non-finite values with a small positive number
    soft_min = np.where(np.isfinite(soft_min), soft_min, 1e-9)
    
    # Ensure the output has the correct shape (100,)
    return soft_min
# EVOLVE-BLOCK-END
