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
    if elapsed > 320.0:
        # Return PI/cost as a safe default
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
    
    n = len(mu)
    
    # Cost-Adjusted LCB with budget-scaled exploration
    # Structural change: scale uncertainty penalty by inverse of remaining budget fraction
    # This forces aggressive exploitation as budget depletes
    
    # Estimate remaining budget
    total_cost_estimate = 5.0 + max(0, _acq_state['call_count'] - 1) * 1.5
    remaining_budget = max(30.0 - total_cost_estimate, 0.5)
    
    # Budget fraction: how much budget is left (1.0 = full, ~0.02 = nearly depleted)
    budget_frac = np.clip(remaining_budget / 30.0, 0.02, 1.0)
    
    # Key structural change: kappa scales with 1/sqrt(budget_frac)
    # When budget is plentiful (budget_frac=1), kappa = base
    # When budget is scarce (budget_frac=0.02), kappa = base * 1/sqrt(0.02) ≈ base * 7
    # Wait - that's backwards. We want LESS exploration when budget is scarce.
    # So kappa should DECREASE as budget depletes.
    # kappa = base * sqrt(budget_frac) was the old approach.
    
    # New approach: use a smooth transition that aggressively reduces exploration
    # as budget approaches zero, but maintains reasonable exploration early on.
    # Use: kappa = base * (budget_frac)^gamma where gamma > 1 makes it drop faster
    
    base_kappa = 0.8
    gamma = 1.5  # steeper drop-off than sqrt
    kappa = base_kappa * (budget_frac ** gamma)
    
    # Additional: if we're very close to finding the optimum (small gap),
    # reduce exploration further
    best_mu = np.min(mu)
    gap = f_best - best_mu
    mu_range = np.max(mu) - np.min(mu)
    if mu_range > 1e-8:
        rel_gap = gap / mu_range
        if rel_gap < 0.1:
            kappa *= 0.3
        elif rel_gap < 0.3:
            kappa *= 0.6
    
    # Compute LCB for minimization
    lcb = mu - kappa * sigma
    
    # Cost adjustment: divide by cost
    scores = -lcb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
