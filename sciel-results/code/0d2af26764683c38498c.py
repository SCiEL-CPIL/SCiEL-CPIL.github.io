"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    # Convert to numpy arrays with float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to ensure finite outputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Check if we're running out of time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a reasonable fallback: prioritize low mu / low cost
        return -mu / cost
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware acquisition: EI per unit cost
    # This balances exploitation (high EI) with efficiency (low cost)
    scores = ei / cost
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in the rastrigin function
    # The bonus is scaled by sigma to favor uncertain regions
    # and dampened by cost to avoid expensive explorations
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine exploitation and exploration
    scores = scores + exploration_bonus
    
    # Final sanitization to ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
