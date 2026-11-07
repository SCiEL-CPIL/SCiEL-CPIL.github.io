import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with cost normalization.
    Samples from the posterior and scores by (sample - f_best) / cost,
    which naturally balances exploration (high sigma) and exploitation (low mu)
    while accounting for evaluation cost.
    
    Args:
        mu: Mean of the GP prediction, shape (2000,)
        sigma: Standard deviation of the GP prediction, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each point, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: draw a sample from the posterior at each candidate
    # Using a fixed seed based on the current state for reproducibility within the call
    # The randomness comes from the fact that each call gets a different random draw
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(size=mu.shape)
    
    # Score: we want to maximize (f_best - sample) / cost
    # This is equivalent to minimizing sample / cost, but we invert for "higher is better"
    # Actually, for minimization: we want low sample values, so (f_best - sample) is good
    # Dividing by cost makes expensive points less attractive
    scores = (f_best - samples) / cost_safe
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
