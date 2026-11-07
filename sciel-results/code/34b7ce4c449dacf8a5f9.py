import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_ACQ_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed function value
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = more promising)
    """
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 325:  # safety margin of 5s
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero and negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0)
    
    # Cost-aware EI: EI per unit cost
    cost_aware_ei = ei / cost_safe
    
    # Add a small exploration bonus for low-cost regions
    # This slightly prefers cheaper points without dominating the EI term
    exploration_bonus = -0.05 * (cost_safe - 1.0)
    
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
