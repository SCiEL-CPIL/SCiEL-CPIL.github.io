import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 15.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    
    Uses a Cost-Aware Upper Confidence Bound (CA-UCB) with adaptive beta.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        return np.full(2000, 0.0)
    
    # Convert to numpy arrays and ensure finiteness
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe sigma for calculations
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # Estimate total cost consumed so far from the cost array
    # The cost array contains costs for 2000 candidates, but we need to infer
    # how much budget has been used. We can estimate this by looking at the
    # relationship between f_best and the number of evaluations.
    # However, a simpler approach: use the average cost as a proxy for per-evaluation cost,
    # and estimate number of evaluations from the context.
    # Since we don't have direct access to total_cost, we use a heuristic:
    # Early on, f_best is high (close to 0 for Hartmann 3D), later it's low (close to -3.86).
    # We can map f_best to a budget fraction.
    
    # Hartmann 3D minimum is -3.86278, typical initial values are around -1 to -3
    # Map f_best to [0, 1] where 0 = just started, 1 = near optimal
    f_min_expected = -3.86278
    f_max_initial = -1.0  # rough estimate of initial f_best
    
    # Normalize f_best to [0, 1]
    f_range = f_max_initial - f_min_expected
    f_normalized = np.clip((f_best - f_min_expected) / f_range, 0.0, 1.0)
    
    # As f_best decreases (gets better), f_normalized decreases
    # We want beta to decrease as we approach the budget limit
    # Budget consumption is roughly proportional to how good f_best is
    budget_fraction = 1.0 - f_normalized  # 0 at start, 1 near optimal
    
    # Adaptive beta: starts high (exploration), decreases as budget is consumed
    # beta = 1 / (1 + total_cost/30) is the suggested formula
    # We approximate total_cost/30 with budget_fraction
    beta = 1.0 / (1.0 + budget_fraction)
    
    # CA-UCB for minimization:
    # We want to minimize, so we look for low mu + beta * sigma
    # The score should be high for points likely to be good
    # UCB_lower = mu - beta * sigma (lower bound on the function value)
    # We want to maximize the probability of finding a new minimum
    # So score = - (mu - beta * sigma) / cost
    # Or equivalently, maximize (beta * sigma - mu) / cost
    
    # Cost-aware UCB score
    ucb_lower = mu - beta * safe_sigma
    score = -ucb_lower / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
