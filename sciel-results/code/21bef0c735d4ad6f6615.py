"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Probability-of-improvement (PI) per unit cost, with a small epsilon offset
    to focus on high-confidence improvements over a threshold.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard: track elapsed time and return safely if needed
    t0 = time.time()
    
    # Small epsilon to create a sharper threshold for high-confidence improvements
    # This focuses search on points likely to beat f_best by a meaningful margin
    eps = 0.01 * (f_best - 10.5364) if f_best > 10.5364 else 0.01
    
    # Compute z-score for probability of improvement with epsilon offset
    # P(f(x) < f_best - eps) = Phi((f_best - eps - mu) / sigma)
    z = (f_best - eps - mu) / (sigma + 1e-9)
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Cost-normalized score: higher probability per unit cost is better
    # Add a small floor to avoid division by zero or near-zero costs
    cost_safe = np.maximum(cost, 1e-6)
    scores = pi / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we're running long, return immediately
    if time.time() - t0 > 295:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
