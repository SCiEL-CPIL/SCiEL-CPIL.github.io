"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: cost-normalized EI with adaptive UCB exploration.
    
    Uses Expected Improvement as the primary exploitation signal, adds an
    adaptive UCB term for exploration, and normalizes by evaluation cost.
    The UCB weight adapts based on how much uncertainty remains relative
    to the current best value.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -30.0, 30.0)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB weight:
    # When uncertainty is high relative to the gap, explore more.
    # When we're close to the optimum (small f_best, low sigma), exploit more.
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Scale: if sigma is large relative to |f_best|, we need more exploration
    # If f_best is close to 0 (optimum), we can exploit more aggressively
    gap_scale = max(abs(f_best), 1e-10)
    
    # Adaptive beta: higher when uncertainty is significant relative to the gap
    # beta ranges roughly from 0.1 (exploit) to 1.0 (explore)
    uncertainty_ratio = min(sigma_mean / gap_scale, 5.0)
    beta = 0.1 + 0.9 * np.tanh(uncertainty_ratio)
    
    # Also consider the spread: if sigma varies a lot, trust high-sigma regions
    sigma_cv = np.std(sigma_safe) / (sigma_mean + 1e-12)
    beta = beta * (1.0 + 0.3 * min(sigma_cv, 2.0))
    
    # Cap beta to avoid over-exploration
    beta = min(beta, 1.5)
    
    ucb_term = beta * sigma_safe
    
    # Combined acquisition before cost normalization
    combined = ei + ucb_term
    
    # Cost-aware: divide by cost to get improvement per unit cost
    score = combined / cost_safe
    
    # Ensure all values are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 280.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
