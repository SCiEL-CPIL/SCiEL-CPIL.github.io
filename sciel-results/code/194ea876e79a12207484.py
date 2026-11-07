"""
Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CA-EI).
This approach favors points with a high expected improvement over the current best,
normalized by their evaluation cost. This balances exploitation (high EI) with
exploration (high sigma) and cost efficiency.
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
    start_time = time.time()
    
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # If mu < f_best, z > 0, high probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is unbounded but typically scales with the function magnitude.
    # We normalize by cost to get cost-per-improvement.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check (should not trigger for N=2000)
    elapsed = time.time() - start_time
    if elapsed > 295:
        pass
        
    return scores
