"""Cost-aware acquisition for three-hump-camel 2D BO.

Structural change: Replace the normalized weighted EI/PI blend with a
cost-aware Expected Improvement (EI) that directly models the trade-off
between expected improvement and evaluation cost. This is a more principled
approach than normalizing separate metrics.

The key insight is that for a fixed budget, we want to maximize the expected
improvement per unit of cost. The cost-aware EI is defined as:
  score = EI(x) / (1 + lambda * cost(x))
where lambda is a tunable parameter that controls the trade-off.

Additionally, we add a small exploration bonus based on the predictive
uncertainty to avoid getting stuck in local optima.
"""
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, with safety margin before 330s limit


def _check_deadline():
    """Return True if we're approaching the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    The score is EI(x) / (1 + lambda * cost(x)) plus a small exploration bonus.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Check deadline - if close, return a simple but fast result
    if _check_deadline():
        # Fast fallback: just use EI without cost adjustment
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: divide by (1 + lambda * cost)
    # Lambda controls how much we penalize expensive evaluations
    # We use a moderate lambda to balance exploration and exploitation
    lambda_cost = 0.5
    
    cost_factor = 1.0 + lambda_cost * cost
    cost_factor = np.maximum(cost_factor, 1e-9)
    
    # Cost-aware EI
    score = ei / cost_factor
    
    # Add a small exploration bonus based on predictive uncertainty
    # This helps avoid getting stuck in regions with low EI but high uncertainty
    # The bonus is scaled to be small relative to typical EI values
    mean_ei = np.mean(ei) + 1e-9
    exploration_bonus = 0.05 * sigma_safe / mean_ei
    score = score + exploration_bonus
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
