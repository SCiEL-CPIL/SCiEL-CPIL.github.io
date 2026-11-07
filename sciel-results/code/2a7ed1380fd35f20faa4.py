# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Probability of Improvement (PI)
    with an adaptive cost exponent that increases as the budget depletes.
    
    Structural idea: PI directly measures P(f(x) < f_best), which aligns with
    the simple regret objective. Unlike EI, PI avoids magnitude skew and is
    more robust in early stages with high uncertainty.
    
    Refinement: Adaptive cost exponent alpha that starts at 1.0 (standard PI/cost)
    and increases to 2.0 (strong cost avoidance) as the remaining budget shrinks.
    This makes the acquisition more aggressive in avoiding expensive evaluations
    late in the run, preserving budget for potentially better points.
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin before 330s limit

    # Increment call count
    _call_count += 1

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple PI/cost
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        cost_safe = np.maximum(cost, eps)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure PI is in [0, 1]
    pi = np.clip(pi, 0.0, 1.0)
    
    # Estimate remaining budget
    # Initial 5 points cost ~7.5 (uniform in [0,1]^6, mean cost = 1 + 0.5 = 1.5 each)
    # Each subsequent point costs ~1.5 on average
    initial_cost_est = 7.5
    avg_cost_per_step = 1.5
    total_cost_est = initial_cost_est + _call_count * avg_cost_per_step
    remaining_budget = max(0.0, 30.0 - total_cost_est)
    
    # Adaptive cost exponent:
    # When budget is plentiful (remaining > 10), alpha = 1.0 (standard PI/cost)
    # When budget is tight (remaining < 5), alpha = 2.0 (strong cost avoidance)
    # Linear interpolation in between
    if remaining_budget > 10.0:
        alpha = 1.0
    elif remaining_budget < 5.0:
        alpha = 2.0
    else:
        alpha = 1.0 + (10.0 - remaining_budget) / 5.0
    
    # Score = PI / (cost ** alpha)
    scores = pi / (cost_safe ** alpha)
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
