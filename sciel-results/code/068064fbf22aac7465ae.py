"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement (EI) for minimization.
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    where Phi is CDF and phi is PDF of standard normal.
    """
    from scipy.stats import norm
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Improvement is positive if mu < f_best
    # If sigma is very small, EI is approx max(0, f_best - mu)
    z = (f_best - mu) / sigma_safe
    
    # Handle cases where z is very large/small to avoid numerical issues
    # Phi(z) ~ 1 for large z, Phi(z) ~ 0 for small z
    # phi(z) is small for large |z|
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    return ei

def _pareto_front_indices(acq_scores, costs):
    """
    Identify indices of points on the Pareto front.
    We want to MAXIMIZE acq_scores and MINIMIZE costs.
    A point is non-dominated if no other point has higher acq_score AND lower cost.
    """
    n = len(acq_scores)
    is_pareto = np.ones(n, dtype=bool)
    
    # Sort by acquisition score descending, then cost ascending for stability
    # This allows us to prune efficiently
    order = np.lexsort((costs, -acq_scores))
    sorted_scores = acq_scores[order]
    sorted_costs = costs[order]
    
    # Keep track of the minimum cost seen so far
    min_cost_so_far = np.inf
    # We iterate from best score to worst score
    for i in range(n):
        idx = order[i]
        # If current cost is less than or equal to min cost seen so far,
        # it is potentially on the Pareto front (since score is >= all previously considered)
        # Actually, since we sorted by score desc, any point with cost < min_cost_so_far
        # is non-dominated by any point with higher score.
        # Points with cost >= min_cost_so_far are dominated by the point that set min_cost_so_far
        # (which has higher or equal score and lower or equal cost).
        
        if sorted_costs[i] <= min_cost_so_far:
            is_pareto[idx] = True
            min_cost_so_far = sorted_costs[i]
        else:
            is_pareto[idx] = False
            
    return is_pareto

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-front based selection on EI.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean predictions from GP.
    sigma : array_like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x_scaled_to_01)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher is better.
    """
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin from 330s limit
    
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values in inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement
    ei_scores = _expected_improvement(mu, sigma, f_best)
    
    # Check time
    if time.time() > deadline:
         # Fallback to simple LCB if EI is too slow (unlikely for 2000 points)
         beta = 2.0
         scores = -mu + beta * sigma
         scores = scores / cost
         return np.where(np.isfinite(scores), scores, -1e10)
    
    # Identify Pareto front: Maximize EI, Minimize Cost
    is_pareto = _pareto_front_indices(ei_scores, cost)
    
    # Construct final scores
    # Points on the Pareto front get a high base score + normalized EI
    # Points off the front get a low score
    # We want the evaluator to pick the best one, so we rank Pareto points by EI primarily,
    # but we also want to distinguish them.
    
    # Normalize EI to [0, 1] for weighting
    ei_min = np.min(ei_scores)
    ei_max = np.max(ei_scores)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.zeros_like(ei_scores)
    else:
        ei_norm = (ei_scores - ei_min) / ei_range
        
    # Base score for Pareto points is 1.0 + ei_norm (so 1.0 to 2.0)
    # Base score for non-Pareto points is very low (e.g., -1.0)
    scores = np.where(is_pareto, 1.0 + ei_norm, -1.0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
