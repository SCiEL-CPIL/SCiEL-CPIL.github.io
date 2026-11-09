import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Samples from the posterior of improvement for each candidate,
    then applies a cost-aware discount to the sampled values.
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin check
    if time.time() - start_time > 290:
        return np.zeros_like(mu)

    # Prevent division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: sample from posterior of improvement
    # For minimization, improvement = f_best - f(x)
    # f(x) ~ N(mu, sigma^2), so improvement ~ N(f_best - mu, sigma^2)
    # Sample one realization per candidate
    rng = np.random.default_rng()
    noise = rng.standard_normal(mu.shape)
    
    # Sampled improvement values
    sampled_improvement = (f_best - mu) + sigma_safe * noise
    
    # Cost-aware discount: multiply by exp(-k * cost)
    # This penalizes high-cost evaluations while preserving the stochastic nature
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    scores = sampled_improvement * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if time.time() - start_time > 295:
        pass
         
    return scores
