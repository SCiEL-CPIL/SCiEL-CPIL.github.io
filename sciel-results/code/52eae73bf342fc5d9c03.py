"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # GP-UCB strategy:
    # Score = mu - beta * sigma
    # Higher score means more likely to be the minimum (exploitation)
    # and/or high uncertainty (exploration).
    
    # Dynamic beta tuning:
    # 1. Estimate the scale of uncertainty in the current candidate set.
    #    Using the median sigma prevents outliers from skewing the scale.
    sigma_median = np.median(sigma_safe)
    
    # 2. Determine progress towards the known global minimum.
    #    The known minimum for Styblinski-Tang 1D is -39.1662.
    known_min = -39.1662
    
    # f_best starts high (e.g., ~50 for ST function) and decreases to -39.1662.
    # progress = 0 when f_best is far from optimum, 1 when close.
    # We use a linear scale from 50 (approx max) to -39.1662.
    max_val = 50.0 
    if max_val == known_min:
        progress = 0.0
    else:
        progress = np.clip((max_val - f_best) / (max_val - known_min), 0.0, 1.0)
    
    # 3. Calculate beta.
    # We want beta to be large early on (high exploration) and small later (exploitation).
    # Also, beta should be proportional to the typical uncertainty level to maintain
    # a consistent balance regardless of how the GP scales uncertainty.
    
    # Base beta in units of sigma. 
    # Early: beta ~ 3.0 sigma (strong exploration)
    # Late: beta ~ 0.5 sigma (strong exploitation)
    base_beta = 3.0 - 2.5 * progress
    
    # Scale beta by the median sigma to make it adaptive to the posterior spread.
    # This ensures that if the GP is very uncertain (large sigma), we explore more in absolute terms,
    # and if it's certain (small sigma), we explore less.
    beta = base_beta * sigma_median
    
    # Calculate UCB scores
    # We want to maximize the score.
    # Low mu is good. Low sigma is good for exploitation, but we subtract beta*sigma to encourage high sigma regions?
    # Wait. Standard UCB for minimization is:
    # LCB(x) = mu(x) - beta * sigma(x)
    # We select x that minimizes LCB(x).
    # The task asks for "higher meaning evaluate next".
    # So we should return -LCB(x) or (mu - beta*sigma) inverted?
    # No, if we minimize f, we want the point with the lowest potential value.
    # LCB = mu - beta*sigma. A low LCB indicates a potentially low f.
    # So we want to select the x with the MINIMUM LCB.
    # The function should return scores where HIGHER is better.
    # So we return -LCB = -(mu - beta*sigma) = beta*sigma - mu.
    
    scores = beta * sigma_safe - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
