"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_DEADLINE_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling and LCB.
    
    Uses Thompson Sampling to draw a sample from the posterior GP,
    combined with a Lower Confidence Bound for deterministic exploration,
    with adaptive cost sensitivity.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    if time.time() - _START_TIME > _TIME_LIMIT - _DEADLINE_MARGIN:
        # Fallback: simple EI/cost, but ensure finite
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e2)
    
    # Thompson Sampling: draw a single sample from the posterior
    # sample = mu + sigma * z, where z ~ N(0, 1)
    z_sample = np.random.standard_normal(len(mu))
    ts_sample = mu + sigma * z_sample
    
    # LCB component: lower confidence bound (we want to minimize)
    # Adaptive beta: scale with maximum sigma to balance exploration
    sigma_max = np.max(sigma)
    beta = 2.0 * sigma_max
    
    lcb = mu - beta * sigma
    
    # Cost penalty: penalize expensive evaluations
    # Normalize cost to [0, 1] range for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Combine Thompson Sampling and LCB
    # For Thompson Sampling: we want to minimize the sampled value, so score = -ts_sample
    # For LCB: we want to minimize the LCB, so score = -lcb
    # Weighted combination: 50% TS, 50% LCB
    ts_score = -ts_sample
    lcb_score = -lcb
    
    # Normalize both scores to comparable scales
    ts_max = np.max(ts_score)
    ts_min = np.min(ts_score)
    if ts_max > ts_min:
        ts_norm = (ts_score - ts_min) / (ts_max - ts_min)
    else:
        ts_norm = np.zeros_like(ts_score)
    
    lcb_max = np.max(lcb_score)
    lcb_min = np.min(lcb_score)
    if lcb_max > lcb_min:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    combined = 0.5 * ts_norm + 0.5 * lcb_norm
    
    # Cost-aware: divide by cost, with a floor to avoid division by near-zero
    cost_safe = np.clip(cost, 1e-6, None)
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
