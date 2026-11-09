# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _safe_return(scores, deadline):
    """Ensure scores are finite and within bounds before returning."""
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, -1e6, 1e6)
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    sigma_safe = sigma + eps
    cost_safe = cost + eps
    
    # Clip f_best to a reasonable range
    f_best_clipped = np.clip(f_best, 0.0, 200.0)
    
    # Compute z-score
    z = (f_best_clipped - mu) / sigma_safe
    
    # Probability of Improvement (PI) for minimization
    pi = norm.cdf(z)
    
    # Cost-aware PI
    scores_pi = pi / cost_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best_clipped - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    scores_ei = ei / cost_safe
    
    # Adaptive blending:
    # Early (high f_best): PI is more robust when GP scale is not well-calibrated
    # Late (low f_best): EI provides sharper exploitation signal
    
    # Weight schedule based on f_best
    # w_pi: high when f_best is large, low when f_best is small
    w_pi = 0.3 + 0.5 * np.exp(-f_best_clipped / 3.0)
    w_ei = 1.0 - w_pi
    
    # Ensure weights are positive
    w_pi = np.maximum(w_pi, 0.1)
    w_ei = np.maximum(w_ei, 0.1)
    
    # Normalize weights
    w_sum = w_pi + w_ei
    w_pi /= w_sum
    w_ei /= w_sum
    
    # Combine scores
    scores = w_pi * scores_pi + w_ei * scores_ei
    
    return _safe_return(scores, deadline)
# EVOLVE-BLOCK-END
