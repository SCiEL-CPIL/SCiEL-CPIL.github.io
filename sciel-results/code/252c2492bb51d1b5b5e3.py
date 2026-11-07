import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are 1D arrays of the same length
    n = len(mu)
    
    # Safety: handle edge cases where sigma is 0 or non-finite
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # 1. Thompson Sampling: Sample from the GP posterior
    # We draw a single sample from N(mu, sigma^2) for all candidates.
    # This implicitly balances exploration (high sigma) and exploitation (low mu).
    z = np.random.standard_normal(n)
    samples = mu + sigma_safe * z
    
    # 2. Cost-Awareness
    # We want to minimize the objective function.
    # Cost is in [1, 2]. Lower cost is better.
    # To make the score "higher is better", we can invert the cost.
    # A common cost-aware metric is Value / Cost.
    # Since we want to minimize Value, we want to maximize (1/Value) or use a penalty.
    # However, simple regret is the primary metric.
    # Let's use the sampled value itself, but penalize high cost.
    # Score = -sampled_value / cost
    # This means:
    # - Lower sampled value (better objective) -> Higher score (less negative)
    # - Lower cost (cheaper) -> Higher score (less negative division)
    
    # To avoid division by zero or negative costs (though cost >= 1), we assume cost > 0.
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    scores = -samples / cost_safe
    
    # 3. Ensure finite outputs
    # If any score is non-finite, replace with a very small number (worst case)
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
