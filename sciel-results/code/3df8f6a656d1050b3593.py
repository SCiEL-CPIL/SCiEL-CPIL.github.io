"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a hybrid strategy combining:
    1. Expected Improvement (EI) for exploitation
    2. Upper Confidence Bound (UCB) for exploration
    3. Knowledge-gradient-like weighting based on uncertainty
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound with adaptive beta
    # Use a larger beta to encourage more exploration
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want low mu and high uncertainty
    # UCB for minimization: we prefer points where mu - beta*sigma is low
    # But actually, for minimization, we want to explore where the lower bound is promising
    # Lower confidence bound: LCB = mu - beta * sigma (we want this to be low)
    lcb = mu - beta * sigma_safe
    
    # Hybrid: combine EI and LCB-based exploration
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = ei.min(), ei.max()
    lcb_min, lcb_max = lcb.min(), lcb.max()
    
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    
    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # For minimization, lower LCB is better, so we invert
    lcb_score = 1.0 - lcb_norm
    
    # Weighted combination: 70% exploitation (EI), 30% exploration (LCB)
    w_ei = 0.7
    w_explore = 0.3
    
    scores = w_ei * ei_norm + w_explore * lcb_score
    
    # Additional boost for points with high uncertainty that are near the best
    # This helps escape local minima
    uncertainty_boost = 0.05 * (sigma_safe / (sigma_safe.max() + 1e-8))
    
    # Proximity to best: prefer points where mu is close to f_best
    proximity = np.exp(-0.5 * ((mu - f_best) / (sigma_safe + 1e-8))**2)
    
    scores += uncertainty_boost * proximity
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we're running out of time, return immediately
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
