"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = None
_DEADLINE_LIMIT = 310.0  # Safety margin before 330s kill

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_LIMIT

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    _init_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Structural change: Probability of Improvement (PI) acquisition
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This directly measures the probability that evaluating x will improve f_best.
    # For smooth functions like Griewank, high-confidence exploitation is effective.
    
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    
    # Add a small exploration component to avoid premature convergence:
    # Scale uncertainty by a small factor so that regions with high sigma
    # get a slight boost, but PI remains the dominant signal.
    # This is structurally different from LCB: it's a bounded [0,1] probability
    # with a small additive exploration term, not a linear confidence bound.
    eps = 0.05
    exploration = eps * (sigma_safe / (np.max(sigma_safe) + 1e-12))
    
    # Combined score: PI + small exploration
    combined = pi + exploration
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if _check_deadline():
        return scores
    
    return scores
# EVOLVE-BLOCK-END
