"""Rewritten acquisition function for cost-aware Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for time tracking
_start_time = time.time()
_DEADLINE = _start_time + 280  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed objective value so far.
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _start_time, _DEADLINE
    
    # Check if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 270:  # Very close to deadline, return quickly
        # Fallback: just return normalized EI without complex computation
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-10)
        scores = ei / cost_safe
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be >= 0 theoretically)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost to get cost-aware score
    cost_safe = np.maximum(cost, 1e-10)
    
    # Base cost-aware EI
    scores = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty relative to cost
    # This helps when EI is zero but uncertainty is high
    # Use a weighted combination: primarily EI/cost, with small sigma/cost term
    uncertainty_bonus = 0.05 * sigma_safe / cost_safe
    
    # Combine: primarily cost-aware EI, with small exploration component
    scores = scores + uncertainty_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
