"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 310.0:
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        z = (f_best - mu_safe) / np.maximum(sigma_safe, 1e-8)
        pi = norm.cdf(z)
        return pi / np.maximum(cost_safe, 1e-8)
    
    # Ensure proper array types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Core: Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Estimate remaining budget to modulate exploration
    total_cost_estimate = 5.0 + max(0, _acq_state['call_count'] - 1) * 1.5
    remaining_budget = max(30.0 - total_cost_estimate, 0.5)
    budget_frac = np.clip(remaining_budget / 30.0, 0.02, 1.0)
    
    # Exploration bonus: scales with uncertainty and remaining budget
    # Use a smooth decay: exploration is strong early, vanishes late
    # The bonus is proportional to sigma * budget_frac^gamma
    gamma = 1.2
    exploration_weight = 0.15 * (budget_frac ** gamma)
    
    # Normalize sigma for the exploration term to keep it in comparable units
    # Use relative uncertainty: sigma / (|f_best| + 1) to make it scale-invariant
    scale_ref = abs(f_best) + 1.0
    rel_sigma = sigma / scale_ref
    
    # Exploration bonus: reward high-uncertainty regions when budget allows
    exploration_bonus = exploration_weight * rel_sigma
    
    # Combine: PI + exploration bonus, both divided by cost
    scores = (pi + exploration_bonus) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
