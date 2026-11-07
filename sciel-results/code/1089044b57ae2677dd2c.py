"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto front selection.
    
    1. Compute Expected Improvement (EI).
    2. Identify the Pareto front of candidates in (EI, Cost) space.
       (Maximize EI, Minimize Cost).
    3. Assign scores to candidates on the Pareto front based on a 
       cost-adjusted EI metric (EI / (1 + Cost)).
    4. Assign very low scores to candidates not on the Pareto front.
    """
    # 1. Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    n = len(ei)
    
    # 2. Identify Pareto Front using vectorized domination check
    # A point i is non-dominated if no other point j dominates it.
    # j dominates i if: ei[j] >= ei[i] AND cost[j] <= cost[i] AND (ei[j] > ei[i] OR cost[j] < cost[i])
    
    ei_col = ei[:, np.newaxis]  # (N, 1)
    ei_row = ei[np.newaxis, :]  # (1, N)
    
    cost_col = cost[:, np.newaxis]  # (N, 1)
    cost_row = cost[np.newaxis, :]  # (1, N)
    
    ei_geq = ei_row >= ei_col
    cost_leq = cost_row <= cost_col
    
    ei_gt = ei_row > ei_col
    cost_lt = cost_row < cost_col
    
    dominates = (ei_geq & cost_leq) & (ei_gt | cost_lt)
    
    is_dominated = np.any(dominates, axis=0)
    pareto_mask = ~is_dominated
    
    # 3. Assign Scores
    scores = np.zeros(n)
    
    if np.any(pareto_mask):
        front_indices = np.where(pareto_mask)[0]
        front_ei = ei[front_indices]
        front_cost = cost[front_indices]
        
        # Use cost-adjusted EI as the score for front points
        # This ensures that among Pareto optimal points, we prefer those
        # with high improvement relative to their cost.
        front_scores = front_ei / (1.0 + front_cost)
        
        # Normalize to [0, 1] range for consistency, though relative order is what matters
        max_score = np.max(front_scores)
        if max_score > 0:
            scores[front_indices] = front_scores / max_score
        else:
            scores[front_indices] = 1.0
            
        # Non-front points get 0
        scores[~pareto_mask] = 0.0
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
