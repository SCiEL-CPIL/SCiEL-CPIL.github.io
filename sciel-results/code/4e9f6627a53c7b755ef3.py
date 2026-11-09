"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a dynamic Lower Confidence Bound (LCB) heuristic,
    inspired by Thompson Sampling's exploration-exploitation balance.
    
    Args:
        mu (np.ndarray): Mean of the GP posterior at candidate points. Shape (N,).
        sigma (np.ndarray): Std dev of the GP posterior at candidate points. Shape (N,).
        f_best (float): The lowest observed objective value so far.
        
    Returns:
        np.ndarray: Scores for each candidate, higher is better. Shape (N,).
    """
    # Ensure inputs are float64 numpy arrays for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division issues in normalization
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Normalize sigma by its median to make the exploration term relative to the current batch's uncertainty level.
    # This helps if the GP variance scale changes or if we are in a region with generally high/low uncertainty.
    med_sigma = np.median(sigma_safe)
    if med_sigma < 1e-8:
        med_sigma = 1e-8
        
    norm_sigma = sigma_safe / med_sigma
    
    # Dynamic exploration weight
    # Scaled by f_best: 
    # - Early (high f_best): High exploration (large beta).
    # - Late (low f_best): Low exploration (small beta), focusing on exploitation.
    beta = f_best
    
    # Calculate Score:
    # 1. (f_best - mu): Expected improvement in mean (exploitation).
    # 2. beta * norm_sigma: Optimistic bonus for uncertainty (exploration).
    score = (f_best - mu) + beta * norm_sigma
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
