"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and UCB with cost normalization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP surrogate at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP surrogate at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB term: explore regions with high uncertainty
    # Use a moderate exploration coefficient
    kappa = 1.5
    ucb = mu + kappa * sigma
    
    # Blend EI and UCB
    # Normalize both to have similar scales
    # EI is in objective units, UCB is also in objective units
    # Blend: 0.6 * EI + 0.4 * (-UCB) since we want to minimize
    # Actually, for minimization, we want low UCB values
    # So we use -UCB as the exploration term
    blended = 0.6 * ei - 0.4 * ucb
    
    # Cost-aware: divide by cost
    scores = blended / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
