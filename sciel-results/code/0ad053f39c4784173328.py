import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses Cost-Weighted Expected Improvement (CW-EI).
    Score = EI(x) / cost(x)
    
    This structural change shifts from a surrogate lower bound (LCB) to
    a direct estimate of the expected reduction in regret (EI), normalized
    by the evaluation cost. This aligns with the resource allocation view
    of maximizing marginal improvement per unit budget.
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure cost is positive and finite
    cost_safe = np.where(cost > 1e-6, cost, 1e-6)
    cost_safe = np.nan_to_num(cost_safe, nan=1.0, posinf=1.0, neginf=1.0)

    # Ensure sigma is non-negative
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    sigma_safe = np.nan_to_num(sigma_safe, nan=1e-9, posinf=1e-9, neginf=1e-9)

    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If sigma is very small, z can be huge.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent overflow in exp/pdf/cdf calculations
    # norm.cdf and norm.pdf handle large values, but clipping is safer for performance
    z = np.clip(z, -10, 10)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_term = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative, but numerical errors can make it slightly negative
    ei_term = np.maximum(ei_term, 0.0)
    
    # Cost-aware scaling
    # Higher score = higher expected improvement per unit cost
    scores = ei_term / cost_safe

    # Handle potential numerical issues (inf/nan)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
