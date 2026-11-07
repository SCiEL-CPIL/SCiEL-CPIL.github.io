"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time
_start_time = time.time()
_DEADLINE = 325.0  # 5 second safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    weighted by inverse cost, with a time guard for safety.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Time guard: if we're approaching the deadline, return a safe result
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        # Return a small positive score for all candidates to avoid invalid results
        return np.ones(len(mu)) * 1e-9
    
    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Avoid division by zero in cost and sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = sigma * (z * phi(z) + Phi(z))
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = sigma_safe * (z * phi_z + Phi_z)
    
    # Use median cost as a reference scale for normalization
    median_cost = np.median(cost_safe)
    cost_normalized = cost_safe / (median_cost + 1e-9)
    
    # Cost-aware score: EI / cost_normalized
    score = ei / cost_normalized
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps avoid premature convergence by giving some credit to uncertain points
    exploration_bonus = 0.1 * sigma_safe / (np.max(sigma_safe) + 1e-9)
    score += exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
