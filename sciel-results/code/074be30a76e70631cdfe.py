# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 280  # Safety margin before 300s limit
_TIMEOUT_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid LCB + EI approach with cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Compute Lower Confidence Bound (LCB)
    # LCB = mu - kappa * sigma (for minimization)
    # We want to maximize (f_best - LCB) = (f_best - mu + kappa * sigma)
    kappa = 1.5  # Exploration-exploitation trade-off
    lcb_score = f_best - mu + kappa * sigma
    
    # Hybrid: combine EI and LCB components
    # Weight them to balance exploitation (EI) and exploration (LCB)
    # EI is in objective units, LCB score is also in objective units
    # Blend: 50% EI + 50% LCB contribution
    hybrid_score = 0.5 * ei + 0.5 * np.maximum(lcb_score, 0.0)
    
    # Cost-aware: divide by cost
    scores = hybrid_score / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > _TIMEOUT_MARGIN:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
