# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
DEADLINE_MARGIN = 30.0

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _elapsed() > DEADLINE_MARGIN:
        # Fallback to simple EI/cost if running out of time
        from scipy.stats import norm
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: Sample from the posterior
    # We use a single sample for all candidates to maintain correlation structure implicitly 
    # (though here we treat them independently, which is standard for this interface)
    # Actually, since we don't have the covariance matrix, we cannot sample correlated points.
    # We sample independent values from N(mu_i, sigma_i^2).
    eps = np.random.randn(len(mu))
    y_sampled = mu + sigma_safe * eps
    
    # Score: Reward for improvement over f_best, penalized by cost
    # Higher score means better improvement per unit cost
    score = (f_best - y_sampled) / (cost + 1e-12)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
