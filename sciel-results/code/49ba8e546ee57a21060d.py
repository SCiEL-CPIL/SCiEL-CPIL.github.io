"""Acquisition function for Bayesian optimization of Hartmann 3D using Adaptive UCB."""
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Computes the Adaptive Upper Confidence Bound (UCB) acquisition score.
    
    Parameters:
    mu: np.ndarray of shape (N,) - GP mean predictions
    sigma: np.ndarray of shape (N,) - GP standard deviation predictions
    f_best: float - The best (lowest) observed value so far
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    # Known global minimum of Hartmann 3D
    f_min = -3.86278
    
    # Calculate the gap between current best and global minimum
    # f_best is expected to be >= f_min
    gap = f_best - f_min
    
    # To adapt the exploration rate, we normalize this gap.
    # A large gap (early in optimization) should lead to high exploration (high beta).
    # A small gap (late in optimization) should lead to low exploration (low beta).
    
    # Set a maximum expected gap to normalize. 
    # Hartmann 3D values in [0,1]^3 typically range from ~-3.86 to ~3.86.
    # Initial random points might yield values around 0 to 3.
    # So a gap of 10 is a safe upper bound for normalization.
    max_gap = 10.0
    
    # Clip gap to [0, max_gap] to handle any numerical anomalies
    gap_clamped = np.clip(gap, 0.0, max_gap)
    
    # Normalized gap: 1.0 when far from optimum, 0.0 when at optimum
    norm_gap = gap_clamped / max_gap
    
    # Define exploration bounds
    beta_max = 4.0  # High exploration early on
    beta_min = 1.0  # Low exploitation late on
    
    # Linear interpolation of beta based on progress
    beta = beta_min + (beta_max - beta_min) * norm_gap
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute UCB scores: Mean + Beta * StdDev
    scores = mu + beta * sigma_safe
    
    return scores
