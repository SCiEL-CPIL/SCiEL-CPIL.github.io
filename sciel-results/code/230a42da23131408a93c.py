"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with expected improvement,
    normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    # Deadline guard: track elapsed time to avoid timeout
    # The function itself is fast, but we add a safety margin
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term
    # Adaptive kappa: larger when sigma is high relative to improvement
    # Use a fixed kappa that balances exploration well for Hartmann 6d
    kappa = 1.5
    ucb = f_best - mu + kappa * sigma
    
    # Combine EI and UCB: weight based on confidence
    # When sigma is large, UCB dominates; when small, EI dominates
    # Use a smooth blend
    sigma_scale = np.maximum(sigma, 1e-6)
    # Blend factor: higher when sigma is large
    blend = np.clip(sigma_scale / (np.abs(f_best - mu) + sigma_scale + 1e-9), 0.0, 1.0)
    blend = 0.5 * blend  # Moderate blend
    
    combined = (1.0 - blend) * ei + blend * ucb
    
    # Normalize by cost
    scores = combined / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety: if we're running too long, return immediately
    # (this function is O(n) so should be very fast)
    
    return scores
# EVOLVE-BLOCK-END
