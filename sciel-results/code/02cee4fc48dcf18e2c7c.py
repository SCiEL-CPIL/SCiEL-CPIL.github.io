"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Kriging Believer with cost normalization for robust multimodal optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _kriging_believer(mu, sigma, f_best, cost, alpha=1.0):
    """
    Kriging Believer acquisition function.
    Maximizes: -mu + alpha * sigma (for minimization)
    Then normalizes by cost to account for evaluation cost.
    
    This approach is often more effective than EI for multimodal functions
    because it directly targets the most promising regions without the
    complex integral calculation of EI.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Kriging Believer: maximize -mu + alpha * sigma
    # This encourages both low predicted values and high uncertainty
    kb = -mu + alpha * sigma_safe
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # when they have similar acquisition values
    cost_safe = np.maximum(cost, 1e-9)
    scores = kb / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Uses Kriging Believer with adaptive alpha based on f_best.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        # Return a simple, fast score to avoid timeout
        cost_safe = np.maximum(cost, 1e-9)
        return (-mu + sigma) / cost_safe
    
    # Adaptive alpha: 
    # - When f_best is high (early), use larger alpha for more exploration
    # - When f_best is low (late), use smaller alpha for more exploitation
    # Rastrigin values: min=0, typical initial f_best ~ 5-20
    if f_best > 10.0:
        alpha = 2.0
    elif f_best > 5.0:
        alpha = 1.5
    elif f_best > 2.0:
        alpha = 1.0
    elif f_best > 1.0:
        alpha = 0.7
    else:
        alpha = 0.5
    
    scores = _kriging_believer(mu, sigma, f_best, cost, alpha=alpha)
    
    return scores
# EVOLVE-BLOCK-END
