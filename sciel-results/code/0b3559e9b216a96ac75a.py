import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a strong penalty for high evaluation costs.
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian Process (2000,)
    sigma (np.ndarray): Standard deviation of the Gaussian Process (2000,)
    f_best (float): Best observed function value
    cost (np.ndarray): Cost of evaluating each candidate (2000,)
    
    Returns:
    np.ndarray: Acquisition scores (2000,)
    """
    # Deadline guard: return a safe default if we're running out of time
    if _check_deadline():
        # Return cost-based scores (prefer low cost) as a safe fallback
        return 1.0 / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI / cost^2
    # This strongly penalizes high-cost evaluations, biasing search
    # toward the central region where the moved minimum is more likely to be.
    # The squared penalty was observed to improve performance over sqrt or linear penalties.
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / (cost_safe * cost_safe)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
