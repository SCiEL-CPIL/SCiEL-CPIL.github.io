"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a budget-scaled LCB approach.
    
    The exploration term is scaled by sqrt(remaining_budget) to dynamically
    shift from exploration to exploitation as the cost budget depletes.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    t_start = _get_start_time()
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        poi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        return np.where(np.isfinite(poi / cost_safe), poi / cost_safe, 0.0)
    
    # Ensure inputs are valid arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Estimate current total cost from the cost array
    # The cost array represents the cost of each candidate, but we need
    # to estimate the current total cost spent so far.
    # We can estimate this from the range of costs observed.
    # Actually, we don't have direct access to the total cost spent.
    # We'll use a heuristic: the mean of the cost array gives us a sense
    # of the typical cost, and we can use the budget (30) to estimate
    # how many evaluations we've done.
    
    # A better approach: use f_best and the distribution of mu/sigma
    # to estimate convergence, and scale exploration accordingly.
    
    # Budget-scaled LCB:
    # score = -mu / cost + beta * sigma / cost
    # where beta is scaled by sqrt(remaining_budget)
    
    # Estimate remaining budget:
    # We don't know the exact total cost, but we can estimate it.
    # If the mean cost per evaluation is ~1.5 (mid-range), and budget is 30,
    # we've done ~20 evaluations. Remaining budget ~ 30 - 20 = 10.
    # A more robust estimate: use the number of low-mu points as a proxy
    # for convergence.
    
    # Compute a convergence indicator:
    # If many points have mu close to f_best, we're converged.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        convergence = 1.0
    else:
        # Fraction of points with mu within 10% of the minimum
        threshold = np.min(mu) + 0.1 * mu_range
        convergence = np.mean(mu < threshold)
    
    # Scale exploration by sqrt(1 - convergence) to shift to exploitation
    # as we converge.
    exploration_scale = np.sqrt(max(0.0, 1.0 - convergence))
    
    # Base beta for LCB
    beta_base = 2.0
    
    # Scale beta by exploration_scale
    beta = beta_base * exploration_scale
    
    # Compute LCB scores (higher is better, so we negate)
    # score = -mu / cost + beta * sigma / cost
    exploitation = -mu / cost_safe
    exploration = beta * sigma_safe / cost_safe
    
    scores = exploitation + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
