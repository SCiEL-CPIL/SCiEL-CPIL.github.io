# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a Hypervolume contribution score based on the Pareto frontier of
    (cost, EI) pairs. Candidates on the Pareto front (min cost, max EI)
    receive a score proportional to the hypervolume they uniquely contribute
    to the dominated region relative to a reference point.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    n = len(mu)
    
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Sort candidates by cost ascending
    sorted_indices = np.argsort(cost)
    sorted_cost = cost[sorted_indices]
    sorted_ei = ei[sorted_indices]
    
    # Identify Pareto Front points
    # A point is on the front if its EI is strictly greater than the max EI
    # of all points with lower cost.
    # Since we sorted by cost, we just check if current EI > running max EI.
    running_max_ei = -np.inf
    front_mask_sorted = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if sorted_ei[i] > running_max_ei:
            front_mask_sorted[i] = True
            running_max_ei = sorted_ei[i]
    
    # Calculate Hypervolume Contribution for each front point
    scores = np.zeros(n)
    
    # Reference cost: slightly above max cost
    c_ref = np.max(cost) + 1.0
    
    front_indices = np.where(front_mask_sorted)[0]
    
    for m in range(len(front_indices)):
        idx_sorted = front_indices[m]
        c_i = sorted_cost[idx_sorted]
        e_i = sorted_ei[idx_sorted]
        
        if m < len(front_indices) - 1:
            idx_next = front_indices[m + 1]
            c_next = sorted_cost[idx_next]
            width = c_next - c_i
        else:
            width = c_ref - c_i
            
        # Contribution is width * height (EI)
        # Map back to original index
        original_idx = sorted_indices[idx_sorted]
        scores[original_idx] = width * e_i
        
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
