import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or invalid z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We sharpen the PI by scaling down the denominator (sigma * 0.7),
    # which increases the contrast between likely improvements and non-improvements.
    # This is beneficial in short budgets where we need to commit to promising regions.
    
    # Compute sharpened z-score
    z = (f_best - mu) / (sigma_safe * 0.7)
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Cost-aware scaling: 
    # The minimum is guaranteed to be in the inner 80% of the domain.
    # Scaled coords in [0.2, 0.8] => mean in [0.2, 0.8] => cost in [1.2, 1.8].
    # We use a Gaussian-like decay centered at cost=1.0 to penalize expensive points,
    # but keep the penalty moderate to avoid excluding the valid minimum region.
    cost_penalty = np.exp(-0.5 * (cost - 1.0))
    
    # Combine PI with cost-aware scaling
    scores = pi * cost_penalty
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
