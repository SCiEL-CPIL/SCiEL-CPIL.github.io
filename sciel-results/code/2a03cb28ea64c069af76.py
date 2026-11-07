# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global state to track total cost across calls
_total_cost = 5 * 1.5  # Approximate initial cost of 5 uniform points in [0,1]^3
_budget = 30.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using an adaptive Lower Confidence Bound (LCB)
    where the exploration parameter beta decays as the budget is consumed.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n_candidates,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (n_candidates,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (n_candidates,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : np.ndarray, shape (n_candidates,)
        Acquisition scores (higher is better).
    """
    global _total_cost
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: inversely proportional to budget consumption
    # beta = 1 / (1 + total_cost / budget)
    # As total_cost approaches budget, beta -> small (exploitation)
    # At start, total_cost is small, beta is large (exploration)
    beta = 1.0 / (1.0 + _total_cost / _budget)
    
    # For minimization, we use LCB: mu - beta * sigma
    # We want to minimize this, so we maximize the negative: -(mu - beta * sigma) = beta * sigma - mu
    # Divide by cost for cost-awareness
    score = (beta * sigma_safe - mu) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Update total cost with the cost of the selected point (the one with max score)
    # We assume the caller will evaluate the best candidate
    best_idx = np.argmax(score)
    _total_cost += cost[best_idx]
    
    return score
# EVOLVE-BLOCK-END
