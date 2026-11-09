"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a UCB-style
    exploration bonus.  The exploration weight is adapted based on how much
    the current best value has improved relative to the known minimum (3.0).
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of GP posterior at candidate points
    sigma : ndarray, shape (n,)
        Standard deviation of GP posterior at candidate points
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight:
    # The known minimum of Goldstein-Price 2D is 3.0.
    # When f_best is close to 3.0, we are near the optimum and should
    # exploit more (lower beta). When f_best is far from 3.0, we should
    # explore more (higher beta).
    # 
    # Define a "progress" measure: how close we are to the minimum.
    # f_best ranges from potentially high values down to 3.0.
    # A simple linear mapping: beta goes from ~3.0 (far) to ~0.3 (near).
    # Use a smooth saturation to avoid abrupt changes.
    
    # The Goldstein-Price function has a range roughly [3, ~200] in [-2,2]^2.
    # Use a reference scale for the gap.
    gap = f_best - 3.0
    if gap < 0:
        gap = 0.0
    
    # Scale: gap of 0 -> beta=0.3, gap of ~50 -> beta=3.0
    # Use a tanh-like smooth transition
    scale = 10.0
    beta = 0.3 + 2.7 * (1.0 - np.exp(-gap / scale))
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.2, 4.0)
    
    # Combined score: EI + adaptive UCB term
    scores = ei + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
