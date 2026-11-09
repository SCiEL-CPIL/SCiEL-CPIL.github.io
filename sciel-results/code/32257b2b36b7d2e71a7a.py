"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using log-cost penalty on LCB.
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: scalar, lowest observed value
    cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE - 2.0:
        return np.zeros_like(mu)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid log of zero or negative
    cost = np.maximum(cost, 1e-6)
    
    # LCB: mu - kappa * sigma
    # We want to maximize this (since lower f is better, we want low mu and low sigma)
    # But we also want to penalize high cost
    # 
    # Using log-space formulation:
    # score = log(mu - kappa * sigma - C) - beta * log(cost)
    # where C is a small constant to keep log argument positive
    #
    # Actually, let's think about this more carefully.
    # We want to minimize f, so we want low values.
    # LCB gives a lower bound on the true value: f_true >= mu - kappa * sigma
    # We want to select points where this lower bound is low (good) and cost is low.
    #
    # In log space: maximize log(lower_bound) - beta * log(cost)
    # But lower_bound can be negative, so we need to be careful.
    #
    # Alternative: use the formulation from the directive:
    # score = log(mu - kappa * sigma) - 0.5 * log(cost)
    # This requires mu - kappa * sigma > 0, which may not hold.
    #
    # Let's use a safer formulation:
    # score = (mu - kappa * sigma) / cost^beta
    # or in log space with offset:
    # score = log(mu - kappa * sigma + offset) - beta * log(cost)
    
    # Let's use a hybrid approach that's numerically stable:
    # Primary: LCB normalized by cost^0.5 (geometric mean style)
    # Fallback: EI/cost if LCB argument is problematic
    
    kappa = 1.5
    
    # Compute LCB
    lcb = mu - kappa * sigma
    
    # For log formulation, we need positive values
    # Find the range of lcb
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    
    # Offset to make all values positive for log
    offset = -lcb_min + 1e-6 if lcb_min < 0 else 1e-6
    lcb_shifted = lcb + offset
    
    # Log-space score: log(lcb + offset) - 0.5 * log(cost)
    score_log = np.log(lcb_shifted) - 0.5 * np.log(cost)
    
    # Also compute standard EI/cost as a fallback
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    score_ei = ei / cost
    
    # Use log-space score as primary
    # Check for non-finite values
    if not np.all(np.isfinite(score_log)):
        # Fall back to EI/cost
        score = score_ei
    else:
        score = score_log
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
