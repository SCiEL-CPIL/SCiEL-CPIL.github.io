import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate a 'Gain' metric for each candidate
    # Using UCB-style gain: Expected Improvement + Exploration Bonus
    # This provides a robust scalar for the 'Benefit' dimension of the Pareto front
    beta = 2.5
    gain = (f_best - mu) + beta * sigma_safe
    
    # Ensure gain is finite
    gain = np.nan_to_num(gain, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Cost is provided
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Identify Pareto-optimal candidates in (Gain, Cost) space
    # We want to Maximize Gain and Minimize Cost
    
    # Reshape for broadcasting: (N, 1) vs (1, N)
    gain_col = gain[:, np.newaxis]  # Gain of candidate i (column)
    cost_col = cost_safe[:, np.newaxis] # Cost of candidate i (column)
    gain_row = gain[np.newaxis, :]   # Gain of candidate j (row)
    cost_row = cost_safe[np.newaxis, :] # Cost of candidate j (row)
    
    # j dominates i if:
    # 1. Gain_j >= Gain_i AND Cost_j <= Cost_i
    # 2. AND (Gain_j > Gain_i OR Cost_j < Cost_i)
    
    weak_dom = (gain_row >= gain_col) & (cost_row <= cost_col)
    strict_dom = (gain_row > gain_col) | (cost_row < cost_col)
    dominates = weak_dom & strict_dom
    
    # A candidate i is dominated if there exists ANY j that dominates it
    # dominates has shape (N, N). dominates[j, i] is True if j dominates i.
    # We want to know for each i, if any j (axis 0) dominates it.
    is_dominated = np.any(dominates, axis=0)
    
    # Pareto front are the non-dominated points
    pareto_mask = ~is_dominated
    
    # Calculate scores
    # For Pareto points, use a scalarization like Gain / Cost
    # For non-Pareto points, assign a very low score
    scores = np.full(gain.shape, -1e9)
    
    if np.any(pareto_mask):
        # Compute score for Pareto points
        scores[pareto_mask] = gain[pareto_mask] / cost_safe[pareto_mask]
        
        # Ensure the non-Pareto scores are lower than the min Pareto score
        min_pareto_score = np.min(scores[pareto_mask])
        scores[~pareto_mask] = min_pareto_score - 1.0
        
    # Handle case where no points are Pareto (should not happen, but safe)
    if not np.any(pareto_mask):
        scores = gain / cost_safe
        
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
