"""Cost-aware acquisition for Ackley 2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Cost-Weighted Expected Improvement (EI/cost).
    EI captures the magnitude of expected improvement, balancing exploration and exploitation.
    Dividing by cost prioritizes cheaper evaluations.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Check deadline
    if _check_deadline():
        # Fallback: Exploitative score based on low mean, penalized by cost
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return fallback_scores
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace NaN/Inf with reasonable defaults
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Since we are minimizing, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # The standard EI formula for minimization is:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * pdf((f_best - mu)/sigma)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    # Replace any remaining NaN/Inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
