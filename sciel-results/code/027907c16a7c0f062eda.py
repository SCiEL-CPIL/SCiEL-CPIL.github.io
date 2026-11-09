"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier hypervolume contribution.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    cost_penalized = np.maximum(cost, 1e-6)
    
    # Identify Pareto Front (Minimize Cost, Maximize EI)
    # Sort indices by cost
    sort_idx = np.argsort(cost)
    sorted_cost = cost[sort_idx]
    sorted_ei = ei[sort_idx]
    
    # Find indices on the Pareto front
    # A point is on the front if its EI is strictly greater than the max EI of all points with lower cost
    front_indices = []
    max_ei_so_far = -np.inf
    
    for i in range(len(sort_idx)):
        if sorted_ei[i] > max_ei_so_far:
            front_indices.append(sort_idx[i])
            max_ei_so_far = sorted_ei[i]
            
    front_indices = np.array(front_indices)
    
    # If the front is too small or EI is negligible, fallback to Cost-Weighted EI
    if len(front_indices) < 10 or np.max(ei) < 1e-6:
        sigma_max = np.max(sigma_safe)
        if sigma_max > 1e-9:
            exploration_bonus = 0.1 * sigma_safe / sigma_max * np.max(ei)
        else:
            exploration_bonus = 0.0
        
        scores = (ei + exploration_bonus) / cost_penalized
    else:
        scores = np.zeros_like(ei)
        
        # Calculate Hypervolume contribution for points on the Pareto front
        # The contribution of a point on the 2D Pareto front (sorted by cost) is
        # the area of the rectangle it uniquely dominates relative to the next point on the front.
        f_costs = cost[front_indices]
        f_eis = ei[front_indices]
        
        # c_next is the cost of the next point on the front, or max(cost) for the last one
        c_nexts = np.empty_like(f_costs)
        if len(f_costs) > 1:
            c_nexts[:-1] = f_costs[1:]
        c_nexts[-1] = np.max(cost)
        
        # HV contribution = width * height
        # Width = c_next - c_current
        # Height = EI_current (since reference EI is 0)
        hv_contribs = (c_nexts - f_costs) * f_eis
        
        scores[front_indices] = hv_contribs
        
        # Add a tiny exploration bonus to break ties and encourage diversity on the front
        scores[front_indices] += 1e-8 * sigma_safe[front_indices]
        
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
