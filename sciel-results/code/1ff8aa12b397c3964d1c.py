"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 285s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        # This is a safe exploitation strategy
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and bounded
    sigma = np.clip(sigma, 1e-6, 1e6)
    
    # Ensure cost is positive and bounded
    cost = np.clip(cost, 1e-6, 1e6)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for very large/small values
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # EI can be negative if mu > f_best; clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # We want to minimize f(x), so UCB = mu - beta * sigma
    # We want to maximize the score, so we use -UCB = beta * sigma - mu
    # Adaptive beta: larger beta when sigma is large relative to the range of mu
    # Simple fixed beta is often robust. Let's use a slightly higher beta for exploration.
    beta = 0.5
    
    ucb = mu - beta * sigma
    score_ucb = -ucb / cost
    
    # Combine EI and UCB
    # Increase exploration weight to help escape local optima in multi-modal landscape
    # Previous was 0.8/0.2. Trying 0.6/0.4.
    w_ei = 0.6
    w_ucb = 0.4
    
    score = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
