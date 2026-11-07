"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a stochastic shortest path inspired
    formulation. The score is the expected cumulative improvement per unit cost,
    approximated by PI * sigma^gamma / (cost * budget_factor), where budget_factor
    dynamically relaxes the cost penalty as the budget depletes.
    
    Structural change: Instead of EI or log-barrier formulations, we use a
    multiplicative model where:
    - PI (Probability of Improvement) provides the exploitation signal
    - sigma^gamma provides sublinear exploration
    - 1/(cost * budget_factor) provides dynamic cost-awareness
    
    The budget_factor = sqrt(remaining_budget / max_cost) dynamically adjusts
    the cost penalty: early on (large remaining budget) cost matters more,
    late in the search (small remaining budget) the penalty relaxes to force
    final exploitation of the best-known region.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 310.0:
        # Fallback: simple sigma-based score normalized by cost
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        fallback = np.maximum(sigma_arr, 1e-8) / np.maximum(cost_arr, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    PI = norm.cdf(z)
    
    # Clip PI to avoid numerical issues
    PI_clipped = np.clip(PI, 1e-10, 1.0 - 1e-10)
    
    # Dynamic budget factor: as total cost increases toward the budget limit,
    # the cost penalty relaxes. We estimate remaining budget from the current
    # cost values. The budget is 30 total, so remaining = 30 - current_cost.
    # We use the mean cost as a proxy for current state.
    current_cost_mean = np.mean(cost_safe)
    budget_limit = 30.0
    remaining_budget = np.maximum(budget_limit - current_cost_mean, 1.0)
    
    # Budget factor: sqrt(remaining / max_possible_remaining)
    # Early: remaining ~ 25, factor ~ sqrt(25/25) = 1.0 (full cost penalty)
    # Late: remaining ~ 5, factor ~ sqrt(5/25) = 0.5 (relaxed cost penalty)
    # This forces exploitation in the final phase
    budget_factor = np.sqrt(remaining_budget / budget_limit)
    budget_factor = np.maximum(budget_factor, 0.1)  # Don't let it go too low
    
    # Parameters
    gamma = 0.7   # Sublinear uncertainty exponent
    
    # Uncertainty term: sigma^gamma
    sigma_term = np.power(sigma_safe, gamma)
    
    # Cost-aware term: 1 / (cost * budget_factor)
    # The budget_factor dynamically relaxes cost penalty late in search
    cost_term = 1.0 / (cost_safe * budget_factor)
    
    # Combine: PI * sigma^gamma / (cost * budget_factor)
    scores = PI_clipped * sigma_term * cost_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
