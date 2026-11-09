import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300 seconds as per task constraint

def acquisition(mu, sigma, f_best, cost):
    # Check if we're close to the deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a simple deterministic fallback if we're running out of time
        # Use Mean Improvement minus cost penalty
        sigma_safe = np.clip(sigma, 1e-9, None)
        # Mean improvement is f_best - mu
        scores = f_best - mu
        cost_norm = cost - 1.0
        scores -= 0.5 * cost_norm
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Thompson Sampling:
    # Approximate the posterior distribution of the function value f(x)
    # f(x) ~ N(mu, sigma^2)
    # We sample a realization from this distribution for each candidate.
    
    # Generate random normal samples
    eps = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled function values
    f_sampled = mu + sigma * eps
    
    # Calculate sampled improvement: f_best - f_sampled
    # Higher improvement is better
    improvement = f_best - f_sampled
    
    # Apply cost-aware penalty
    # Cost ranges from 1 to 2 (1 + mean(x_scaled))
    # cost_norm = cost - 1.0, ranges [0, 1]
    cost_norm = cost - 1.0
    
    # Additive penalty: subtract a portion of the normalized cost
    # This ensures that for any given improvement, lower cost is preferred.
    # For negative improvements (worse than best), lower cost is also preferred (less negative).
    penalty_weight = 0.5
    penalty = penalty_weight * cost_norm
    
    scores = improvement - penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
