import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Uses UCB which balances exploration (sigma) and exploitation (mu) 
    with a tunable parameter kappa. This is a structural change from 
    Expected Improvement (EI) to a different acquisition family.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        mu_arr = np.asarray(mu)
        # Fallback: pure exploitation
        fallback_scores = -mu_arr
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    _call_count += 1
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety clamps
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive Kappa: Start high for exploration, decrease for exploitation.
    # The total budget is ~30 cost. Initial 5 points cost ~5. 
    # Remaining ~25 cost allows for ~10-20 evaluations depending on x.
    # Let's assume max ~20 steps.
    max_steps = 20.0
    # Linear decay from kappa_max to kappa_min
    kappa_max = 3.0
    kappa_min = 0.5
    progress = min(_call_count / max_steps, 1.0)
    kappa = kappa_max - (kappa_max - kappa_min) * progress
    
    # UCB for minimization: 
    # We want to maximize the potential for low values.
    # Score = -mu + kappa * sigma
    ucb_scores = -mu + kappa * sigma_safe
    
    # Cost-awareness: divide by cost
    scores = ucb_scores / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
