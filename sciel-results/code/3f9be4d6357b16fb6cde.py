import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function with cost-aware normalization.
    
    Args:
        mu: GP mean predictions, shape (2000,)
        sigma: GP standard deviations, shape (2000,)
        f_best: best observed function value
        cost: evaluation costs for each candidate, shape (2000,)
    
    Returns:
        scores: acquisition scores, shape (2000,), higher = more desirable
    """
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    sigma = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from the posterior
    # For each candidate, sample a function value from N(mu, sigma^2)
    # We use a single sample for efficiency, but could use multiple for robustness
    
    # Generate one sample per candidate from the posterior
    sample = mu + sigma * np.random.standard_normal(len(mu))
    
    # For minimization, we want to find candidates with low sampled values
    # Cost-aware: divide by cost to prefer cheaper evaluations
    
    # Score = -sampled_value / cost (we want to maximize this)
    # This inherently balances exploration (via the random sample) and cost
    
    scores = -sample / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
