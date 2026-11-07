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
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # z-score for EI computation
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -20.0, 20.0)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight
    # When f_best is large (far from optimum), explore more
    # When f_best is small (close to optimum), exploit more
    # Scale based on the ratio of sigma to f_best
    # If f_best is close to 0, we should be exploiting heavily
    # If f_best is large, we can afford more exploration
    
    # Normalize f_best to get a sense of how far we are
    # The minimum of levy 2d is 0, so f_best >= 0
    # Use a soft scaling: beta = base * (1 + alpha / (1 + f_best))
    # When f_best is small, beta is larger (more exploration)
    # When f_best is large, beta approaches base
    
    # Actually, let's think about this differently.
    # Early on, f_best is large, we want to explore broadly.
    # Late on, f_best is small, we want to exploit.
    # But also, the scale of sigma matters.
    
    # Let's use the ratio of typical sigma to f_best
    sigma_mean = np.mean(sigma_safe)
    
    # If f_best is very small relative to sigma, we're in exploitation mode
    # If f_best is large relative to sigma, we're in exploration mode
    ratio = sigma_mean / (abs(f_best) + 1e-10)
    
    # Map ratio to beta: higher ratio -> more exploration
    # beta ranges from ~0.1 (exploit) to ~1.5 (explore)
    beta = 0.1 + 1.4 * np.tanh(ratio)
    
    # UCB exploration term
    ucb_term = beta * sigma_safe
    
    # Combined acquisition before cost normalization
    combined = ei + ucb_term
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 280.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
