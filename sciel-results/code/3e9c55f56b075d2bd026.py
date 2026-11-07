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
    
    # Cost-weighted LCB with soft cost penalty
    # LCB: mu - beta * sigma (we want to minimize, so lower is better)
    # We use a dynamic beta that increases with uncertainty
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration parameter: larger when uncertainty is high
    # beta scales with the ratio of sigma to the spread of f_best
    spread = np.std(mu) + 1e-9
    beta = 2.0 * (sigma_safe / (spread + 1e-9))
    beta = np.clip(beta, 0.5, 5.0)
    
    # LCB score: we want to minimize mu - beta*sigma
    lcb = mu - beta * sigma_safe
    
    # Cost penalty: use a soft penalty that doesn't completely dominate
    # cost ranges from 1 to 2, so we use a power penalty
    cost = np.maximum(cost, 1e-6)
    cost_penalty = 0.3 * np.log(cost + 1.0)  # Soft logarithmic penalty
    
    # Combine: lower lcb is better, lower cost penalty is better
    # We want higher score = evaluate next
    scores = -lcb - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
