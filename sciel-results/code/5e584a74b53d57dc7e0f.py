import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses a hybrid of LCB (exploration) and EI (exploitation) with 0.7/0.3 weighting.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Fast deterministic fallback: LCB with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.5 * sigma_safe)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # LCB for minimization: lower is better, so we negate for "higher score = better"
    # LCB = mu - beta * sigma, we want to maximize this (i.e., minimize mu - beta*sigma)
    # So score_lcb = -(mu - beta * sigma) = beta * sigma - mu
    beta = 2.0  # Exploration parameter
    lcb_score = beta * sigma - mu
    
    # Expected Improvement for minimization
    z = (mu - f_best) / sigma
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.maximum(EI, 0.0)
    
    # Hybrid: 0.7 * LCB + 0.3 * EI
    # This balances global exploration (LCB) with local refinement (EI)
    scores = 0.7 * lcb_score + 0.3 * EI
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
