"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: we must finish well before the 330s wall-clock limit.
# The acquisition function is called repeatedly; we track total elapsed time.
_acq_start_time = None

def _elapsed():
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    return time.time() - _acq_start_time

def _time_remaining():
    return 330.0 - _elapsed() - 10.0  # 10s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled) where x_scaled in [0,1]^2.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher = more promising to evaluate next.
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a hybrid acquisition: combine EI with an exploration bonus
    # that accounts for cost. We want to balance exploitation (low mu)
    # and exploration (high sigma) while penalizing high cost.
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    # We divide by cost to get improvement per unit cost
    ei_per_cost = ei / cost
    
    # Add a small exploration term that favors points with high uncertainty
    # relative to the scale of the problem. The Levy function has a minimum of 0
    # and can be quite large, so we scale by f_best to make it adaptive.
    # Use a soft exploration bonus: sigma / (sigma + |f_best| + 1)
    # This gives a bounded bonus in [0, 1) that is larger when uncertainty is high
    # relative to the current best value.
    if f_best > 0:
        explore_bonus = sigma / (sigma + f_best + 1.0)
    else:
        # If f_best is near 0, we're close to the optimum; reduce exploration
        explore_bonus = sigma / (sigma + 1.0)
    
    # Combine: primary signal is EI per cost, with a weighted exploration bonus
    # The weight on exploration should decrease as we get closer to the optimum
    # and as budget is consumed. We don't have explicit budget info here, but
    # cost gives a hint: lower cost points are "cheaper" to explore.
    
    # Weight: start with 0.3 for exploration, but scale down if f_best is small
    if f_best > 1.0:
        w_explore = 0.3
    elif f_best > 0.1:
        w_explore = 0.15
    else:
        w_explore = 0.05
    
    scores = ei_per_cost + w_explore * explore_bonus / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
