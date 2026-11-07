"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition function normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : np.ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Dynamic exploration coefficient:
    # Use the spread of mu relative to f_best to adapt beta.
    # When mu is close to f_best, explore more; when far, exploit.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-9:
        mu_range = 1e-9
    
    # Beta starts moderate and increases with uncertainty relative to improvement
    # Use a simple heuristic: beta proportional to std of sigma
    sigma_mean = np.mean(sigma)
    beta = 2.0 + 3.0 * (sigma_mean / (abs(f_best) + 1e-9))
    
    # Cap beta to avoid excessive exploration
    beta = np.clip(beta, 1.0, 10.0)
    
    # LCB: we want to minimize, so we score by -mu + beta * sigma
    # Higher score = lower predicted value + more uncertainty
    lcb_score = -mu + beta * sigma
    
    # Normalize by cost: prefer candidates that give good LCB per unit cost
    # cost is in [1, 2] since x scaled to [0,1] means mean(x_scaled) in [0,1]
    # so cost = 1 + mean(x_scaled) in [1, 2]
    score = lcb_score / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
