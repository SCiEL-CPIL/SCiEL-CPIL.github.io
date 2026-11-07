import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling variant
    for Bayesian optimization of Ackley 2D.
    """
    t_start = time.time()
    deadline = t_start + 290.0  # safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)

    # Check if we're running out of time
    if time.time() > deadline - 5.0:
        # Fallback: simple LCB
        lcb = -mu + sigma
        score = lcb / cost
        score = np.where(np.isfinite(score), score, -1e6)
        return score

    # Thompson Sampling: sample a scalar offset from Gaussian
    # Mean at f_best, std scales with max uncertainty
    max_sigma = np.max(sigma) if len(sigma) > 0 else 0.0
    
    # If max_sigma is very small, we're confident; exploit more
    # If max_sigma is large, explore more
    offset_std = max_sigma * 0.5 + 1e-6
    
    # Sample the offset
    offset = f_best + np.random.normal(0, offset_std)
    
    # Score: maximize -(mu - offset) / cost
    # This is equivalent to maximizing (offset - mu) / cost
    # Higher scores = better candidates to evaluate next
    score = (offset - mu) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
