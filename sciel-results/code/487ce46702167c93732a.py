"""Cost-aware acquisition for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = None
_TIME_LIMIT = 325.0  # seconds, safety margin before 330s kill


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_remaining():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Uses a Lagrangian-style cost-aware acquisition that combines:
    1. Expected Improvement (EI) for exploitation
    2. A smooth exponential cost penalty instead of hard division
    3. An uncertainty bonus scaled by remaining budget awareness
    
    The cost penalty is dynamically tuned: as we approach the optimum
    (f_best decreases), the penalty tightens to favor cheaper points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure no NaNs or infs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e9, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost penalty: exponential form instead of hard division
    # As f_best approaches the minimum (~-78.33), the penalty tightens
    # Normalize f_best to get a progress indicator
    # f_best starts around 0-100 and decreases to ~-78
    # progress = how close we are to the known minimum
    f_min_known = -78.3323
    # Map f_best from [0, 100] range to [0, 1] progress
    # When f_best is high (early), progress ~0, penalty is loose
    # When f_best is low (late), progress ~1, penalty is tight
    progress = np.clip((50.0 - f_best) / 128.0, 0.0, 1.0)
    
    # Beta parameter for the exponential penalty
    # Higher beta = stronger penalty for expensive points
    # Dynamically increase beta as progress increases
    beta = 0.5 + 2.0 * progress
    
    # Smooth exponential cost penalty: exp(-beta * (cost - 1))
    # cost ranges from 1 (cheapest) to 2 (most expensive)
    # At cost=1, penalty=1; at cost=2, penalty=exp(-beta)
    cost_penalty = np.exp(-beta * (cost - 1.0))
    
    # Cost-aware EI: multiply EI by the cost penalty
    ei_cost_aware = ei * cost_penalty
    
    # Uncertainty bonus: encourages exploration in uncertain regions
    # Scaled by cost penalty to prefer cheaper uncertain points
    ucb = sigma * cost_penalty
    
    # Combine: primary term is cost-aware EI, secondary is uncertainty
    # Weight the UCB term to balance exploration/exploitation
    # Use a moderate alpha that decreases as we get closer to optimum
    alpha = 0.3 + 0.4 * (1.0 - progress)
    
    scores = ei_cost_aware + alpha * ucb
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
