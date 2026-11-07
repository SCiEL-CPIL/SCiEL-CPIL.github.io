import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Cost is in [1, 2] (1 + mean of x scaled to [0,1])
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Structural Change: Rank-based Pareto Front Selection ---
    
    # 1. Identify Pareto Front (Non-dominated set)
    # Objective 1: Maximize EI
    # Objective 2: Minimize Cost
    
    N = len(ei)
    
    # Vectorized dominance check
    # D[i, j] is True if i dominates j
    # i dominates j if:
    # ei[i] >= ei[j] and cost[i] <= cost[j] and (ei[i] > ei[j] or cost[i] < cost[j])
    
    ei_i = ei[:, None]
    ei_j = ei[None, :]
    cost_i = cost[:, None]
    cost_j = cost[None, :]
    
    non_inferior = (ei_i >= ei_j) & (cost_i <= cost_j)
    strict = (ei_i > ei_j) | (cost_i < cost_j)
    dominates = non_inferior & strict
    
    # A point is on the Pareto front if it is NOT dominated by any other point
    # sum(dominates, axis=0) gives count of points dominating j
    # If count is 0, j is non-dominated
    dom_count = np.sum(dominates, axis=0)
    is_pareto = (dom_count == 0)
    
    # 2. Score Assignment
    scores = np.zeros_like(ei, dtype=np.float64)
    
    if np.any(is_pareto):
        # Extract Pareto front points
        p_ei = ei[is_pareto]
        p_cost = cost[is_pareto]
        
        # Normalize EI and Cost within the front
        # We want to score high for High EI and Low Cost
        
        ei_min = np.min(p_ei)
        ei_max = np.max(p_ei)
        cost_min = np.min(p_cost)
        cost_max = np.max(p_cost)
        
        # Avoid division by zero
        ei_range = ei_max - ei_min
        if ei_range < 1e-9:
            ei_range = 1.0
            
        cost_range = cost_max - cost_min
        if cost_range < 1e-9:
            cost_range = 1.0
            
        norm_ei = (p_ei - ei_min) / ei_range
        norm_cost = (p_cost - cost_min) / cost_range
        
        # Composite Score: Weighted sum
        # We prioritize EI (exploitation) but penalize Cost.
        # If all points have same EI, pick lowest cost.
        # If all points have same cost, pick highest EI.
        w_ei = 0.8
        w_cost = 0.2
        
        front_scores = w_ei * norm_ei + w_cost * (1.0 - norm_cost)
        
        # Assign these scores to the corresponding indices
        # We need to map front_scores back to the original indices
        # is_pareto is a boolean mask
        scores[is_pareto] = front_scores
        
        # To ensure the "rank" aspect and distinctness, we can add a tiny epsilon
        # based on the rank in the front to break ties perfectly if needed,
        # but float precision of the weighted sum is usually sufficient.
        
        # However, to strictly follow "proportional to rank", we could sort them.
        # But the weighted sum is a standard scalarization of the Pareto front.
        
        # Let's boost the front scores to be clearly higher than non-front points (which are 0)
        # Since EI is usually > 0, and we normalized to [0,1], the scores are in [0, 1].
        # We can scale them up to ensure they are the argmax.
        scores[is_pareto] *= 10.0
        
    else:
        # Fallback if no Pareto points found (should not happen unless N=0 or NaNs)
        # Use simple EI / Cost
        scores = ei / cost
        
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
