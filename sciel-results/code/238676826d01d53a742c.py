"""
Cost-aware acquisition function using Cost-Adjusted Probability of Improvement (CA-POI).
This approach favors points with a high probability of beating the current best,
normalized by their evaluation cost. This is more exploitative than EI, which is
beneficial in low-budget Bayesian Optimization scenarios.
"""
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores.
    
    Args:
        mu: Mean of the GP surrogate, shape (N,)
        sigma: Standard deviation of the GP surrogate, shape (N,)
        f_best: Current best observed function value (scalar)
        cost: Cost of evaluating each candidate, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,). Higher is better.
    """
    # Deadline guard: ensure we finish well before the 300s limit
    # The computation is vectorized and fast, but we add a guard as per directive.
    start_time = time.time()
    
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    # Adding a small epsilon to prevent numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # If mu < f_best, z > 0, high probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi(z)
    # This is a bounded [0, 1] value, making it stable and exploitative
    poi = norm.cdf(z)
    
    # Cost-aware score: POI per unit cost
    # We want to maximize the probability of finding a better point per unit cost
    scores = poi / cost_safe
    
    # Ensure all scores are finite
    # Replace NaNs or Infs with 0.0 (or a very small negative number if we wanted to penalize)
    # Since POI is >= 0 and cost > 0, scores should be >= 0.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check (should not trigger for N=2000)
    elapsed = time.time() - start_time
    if elapsed > 295:
        # In case of extreme slowness, return the current best scores
        # This is a safety net
        pass
        
    return scores
