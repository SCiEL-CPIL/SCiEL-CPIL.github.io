"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time inside acquisition and bail out early if needed.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.zeros(len(mu))
    
    n = len(mu)
    
    # Convert to numpy arrays and sanitize
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # avoid numerical underflow/overflow
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) for minimization
    xi = 0.01
    lcb = mu - xi * sigma
    
    # Normalize both to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    
    ei_range = ei_max - ei_min
    lcb_range = lcb_max - lcb_min
    
    if ei_range < 1e-10:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    if lcb_range < 1e-10:
        lcb_norm = np.ones_like(lcb)
    else:
        lcb_norm = (lcb - lcb_min) / lcb_range
    
    # LCB lower is better, so invert for scoring
    lcb_score = 1.0 - lcb_norm
    
    # Combine: 70% EI, 30% LCB
    combined = 0.7 * ei_norm + 0.3 * lcb_score
    
    # Cost-aware: divide by cost
    scores = combined / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE:
        return np.zeros(n)
    
    return scores
# EVOLVE-BLOCK-END
