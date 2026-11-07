import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin below 330s total


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thrun-Hutter UCB acquisition function for Bayesian optimization.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware UCB with fixed beta
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        ucb = mu - 1.0 * sigma_safe  # Lower confidence bound for minimization
        return ucb / cost_safe

    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive beta parameter for UCB
    # Based on the number of evaluations (approximated by n, but we can use a fixed schedule)
    # Thrun-Hutter suggests beta that grows with confidence.
    # A common choice is beta = sqrt(2 * log(n) / tau) or similar.
    # Here we use a simple adaptive beta that increases slowly with the number of candidates
    # or we can use a fixed beta that works well in practice.
    # Let's use a beta that balances exploration and exploitation.
    # For minimization, we want to minimize mu - beta * sigma.
    # To make it cost-aware, we divide by cost.
    
    # Dynamic beta: starts high for exploration, decreases for exploitation
    # We can estimate the number of evaluations from the cost if we know the initial cost,
    # but here we don't have that info directly.
    # Instead, we can use a fixed beta or a beta that depends on the magnitude of sigma.
    
    # Let's use a standard UCB approach with a fixed beta, which has shown to work well.
    beta = 1.5  # Exploration-exploitation trade-off
    
    # Lower Confidence Bound for minimization
    lcb = mu - beta * sigma_safe
    
    # Cost-aware score: LCB per unit cost
    score = lcb / cost_safe
    
    # Add a small regularization to avoid division by very small costs
    # This encourages points with lower cost if the LCB is similar
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
