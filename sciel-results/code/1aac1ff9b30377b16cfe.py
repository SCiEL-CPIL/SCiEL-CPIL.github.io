"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier hypervolume contribution
    with smooth soft-min surrogate for improved gradient flow and exploration balance.
    
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
    
    # Normalize cost to [0, 1] for better conditioning
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Normalize EI to [0, 1] for better conditioning
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
    
    # Compute smooth surrogate for the Pareto front using log-sum-exp
    # We want to find points that are on the "smoothed" Pareto front
    # where we trade off cost and EI
    
    # For each point, compute the probability that it would be selected
    # under a soft-min of (cost, -EI) with temperature
    # Lower cost and higher EI are better
    
    # Use a temperature parameter to control smoothness
    temperature = 0.1
    
    # Compute the "badness" score: high cost is bad, low EI is bad
    # We want to minimize cost and maximize EI
    # So we minimize (cost_norm) and maximize (ei_norm)
    # Equivalently, minimize (cost_norm - ei_norm)
    
    # But we want to identify the Pareto front more explicitly
    # Use the approach: for each point, compute how many other points
    # dominate it (have lower cost AND higher EI)
    
    n = len(mu)
    
    # Vectorized dominance counting would be O(n^2) which is too slow for n=2000
    # Instead, use a different approach: compute a smooth score based on
    # the rank of each point in the 2D (cost, EI) space
    
    # Sort by cost
    sort_idx = np.argsort(cost_norm)
    sorted_cost = cost_norm[sort_idx]
    sorted_ei = ei_norm[sort_idx]
    
    # Find the Pareto front (points where EI increases as cost increases)
    # A point is on the front if no other point has both lower cost and higher EI
    
    # Efficient approach: scan from low cost to high cost, track max EI seen
    # A point is on the front if its EI is greater than all previous points' EI
    
    on_front = np.zeros(n, dtype=bool)
    max_ei_so_far = -np.inf
    
    for i in range(n):
        idx = sort_idx[i]
        if sorted_ei[i] > max_ei_so_far:
            on_front[idx] = True
            max_ei_so_far = sorted_ei[i]
    
    # For points on the front, compute hypervolume contribution
    # For points off the front, assign a lower score based on how close they are to the front
    
    scores = np.zeros(n)
    
    if np.any(on_front):
        front_idx = np.where(on_front)[0]
        front_costs = cost_norm[front_idx]
        front_eis = ei_norm[front_idx]
        
        # Sort front points by cost
        front_sort = np.argsort(front_costs)
        front_costs_sorted = front_costs[front_sort]
        front_eis_sorted = front_eis[front_sort]
        front_idx_sorted = front_idx[front_sort]
        
        # Compute hypervolume contribution for each front point
        # The contribution is the area of the rectangle it uniquely dominates
        # Width: distance to next point's cost (or 1.0 for the last)
        # Height: EI value (since reference is 0)
        
        hv_contribs = np.zeros(len(front_idx_sorted))
        if len(front_idx_sorted) > 1:
            widths = np.diff(np.concatenate([front_costs_sorted, [1.0]]))
            # Each point's contribution is width * height
            # But we need to be careful: the width for point i is the cost of point i+1 minus cost of point i
            # And the height is the EI of point i
            hv_contribs[:-1] = widths[:-1] * front_eis_sorted[:-1]
            hv_contribs[-1] = widths[-1] * front_eis_sorted[-1]
        
        # Assign hypervolume contributions to front points
        scores[front_idx_sorted] = hv_contribs
        
        # For off-front points, compute a smooth score based on their proximity to the front
        # Use the minimum "distance" to the Pareto front in a normalized sense
        
        off_front_idx = np.where(~on_front)[0]
        
        if len(off_front_idx) > 0:
            # For each off-front point, find the closest front point
            # and assign a score based on how much improvement it would need
            # to be on the front
            
            # Simple approach: for each off-front point, find the front point
            # with the lowest cost that has higher EI, or the front point
            # with the highest EI that has lower cost
            
            # Even simpler: assign a score proportional to (1 - normalized_distance_to_front)
            # where distance is measured in the (cost, EI) space
            
            # Compute the "gap" for each off-front point: how much EI it lacks
            # compared to the best EI at its cost level or below
            
            # For each off-front point, find the max EI among front points with cost <= its cost
            # If that max EI > its EI, it's dominated
            # The score should be low
            
            # Use a smooth approximation: for each point, compute the probability
            # that it would be on the front under a slightly perturbed distribution
            
            # Simpler: just assign a small score based on EI/cost ratio
            # to ensure some exploration off the front
            
            off_ei = ei_norm[off_front_idx]
            off_cost = cost_norm[off_front_idx]
            
            # Score for off-front points: small fraction of the min front score
            # weighted by EI/cost to encourage some exploration
            min_front_score = np.min(scores[front_idx_sorted]) if len(front_idx_sorted) > 0 else 0.0
            off_scores = 0.1 * min_front_score * off_ei / (off_cost + 1e-6)
            
            scores[off_front_idx] = off_scores
    
    # Add a small exploration bonus based on uncertainty to encourage diversity
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        exploration_bonus = 0.05 * (sigma_safe / sigma_max) * np.max(scores)
        scores += exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
