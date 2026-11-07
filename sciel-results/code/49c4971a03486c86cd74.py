"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 330.0  # seconds
_MARGIN = 5.0      # safety margin


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with cost normalization.
    
    mu: (2000,) predicted mean
    sigma: (2000,) predicted std (in objective units)
    f_best: scalar, best observed value so far
    cost: (2000,) cost of evaluating each candidate = 1 + mean(x_scaled_to_[0,1])
    
    Returns: (2000,) acquisition scores, higher = more desirable to evaluate next.
    """
    # Check deadline; if we're close, return a safe fallback
    if _time_left() < 0.5:
        # Fallback: return UCB without cost normalization (fast path)
        return mu - 0.5 * sigma  # exploit-leaning, finite
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB component: explore where uncertainty is high, exploit where mean is low
    # Use a moderate kappa for balanced exploration/exploitation
    kappa = 1.5
    ucb = mu + kappa * sigma
    
    # Cost normalization: divide by cost^gamma to favor cheap evaluations
    # Higher gamma strongly penalizes expensive regions
    gamma = 2.5
    cost_safe = np.maximum(cost, 1e-6)
    cost_factor = cost_safe ** gamma
    
    # Combine: UCB / cost^gamma
    scores = ucb / cost_factor
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
