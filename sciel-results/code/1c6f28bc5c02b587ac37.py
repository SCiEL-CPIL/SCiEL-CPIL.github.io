"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier selection over
    improvement potential and uncertainty-per-cost.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        return 1.0 / (np.maximum(cost, 1e-9))
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Two independent metrics for Pareto selection:
    # 1. Improvement potential: f_best - mu (higher is better)
    # 2. Uncertainty per cost: sigma / cost (higher is better - efficient exploration)
    
    improvement = f_best - mu
    uncertainty_per_cost = sigma_safe / cost_safe
    
    # Normalize both metrics to [0, 1] range for fair comparison
    imp_min, imp_max = np.min(improvement), np.max(improvement)
    upc_min, upc_max = np.min(uncertainty_per_cost), np.max(uncertainty_per_cost)
    
    imp_range = imp_max - imp_min + 1e-9
    upc_range = upc_max - upc_min + 1e-9
    
    imp_norm = (improvement - imp_min) / imp_range
    upc_norm = (uncertainty_per_cost - upc_min) / upc_range
    
    # Pareto dominance check:
    # Point i dominates point j if imp_i >= imp_j AND upc_i >= upc_j
    # and at least one inequality is strict.
    # 
    # For efficiency with 2000 points, we use a vectorized approach:
    # A point is non-dominated if no other point has both metrics >= it
    # (with at least one strictly greater).
    
    # Efficient Pareto frontier identification:
    # Sort by improvement descending, then scan for points that improve on
    # the best uncertainty_per_cost seen so far.
    
    # Sort indices by improvement (descending)
    order = np.argsort(-imp_norm)
    sorted_imp = imp_norm[order]
    sorted_upc = upc_norm[order]
    
    # A point is on the Pareto frontier if its upc_norm is strictly greater
    # than all upc_norm values of points with higher or equal improvement.
    # We track the running maximum of upc_norm.
    
    # Initialize: all points are potentially on the frontier
    is_pareto = np.zeros(len(mu), dtype=bool)
    
    # Process in order of decreasing improvement
    # A point is non-dominated if its upc is >= all upc of points with higher improvement
    # (since we process in decreasing improvement order, we just need upc >= running max)
    
    running_max_upc = -1.0
    for idx in order:
        if sorted_upc[np.where(order == idx)[0][0]] >= running_max_upc - 1e-12:
            is_pareto[idx] = True
            running_max_upc = max(running_max_upc, sorted_upc[np.where(order == idx)[0][0]])
    
    # This O(n log n) approach may be slow for 2000 points in a loop.
    # Let's use a vectorized approach instead.
    
    # Vectorized Pareto: point i is dominated if exists j with imp_j >= imp_i and upc_j >= upc_i
    # and (imp_j > imp_i or upc_j > upc_i).
    # 
    # Simpler: point i is non-dominated if it has the max imp among points with upc >= upc_i,
    # OR it has the max upc among points with imp >= imp_i, etc.
    #
    # Most efficient vectorized check for 2D:
    # Sort by imp descending. A point is Pareto-optimal if its upc is strictly greater
    # than the max upc of all points before it in the sorted order (i.e., points with higher imp).
    
    order = np.argsort(-imp_norm)
    sorted_upc = upc_norm[order]
    
    # Running max of upc from the start (highest imp) to each position
    # cummax gives the max upc seen so far in the sorted order
    running_max = np.maximum.accumulate(sorted_upc)
    
    # A point at position i in sorted order is Pareto-optimal if its upc equals the running max
    # at that position (i.e., it sets a new record for upc among points with imp >= its imp)
    # We use a small tolerance to handle floating point issues.
    
    is_pareto_sorted = np.isclose(sorted_upc, running_max, rtol=1e-9, atol=1e-12)
    
    # Map back to original indices
    is_pareto = np.zeros(len(mu), dtype=bool)
    is_pareto[order] = is_pareto_sorted
    
    # Score: Pareto-optimal points get a high base score, non-dominated points get 0
    # Among Pareto-optimal points, rank by a combined score to break ties
    # Use the sum of normalized metrics as a tie-breaker within the Pareto set
    
    combined = imp_norm + upc_norm
    
    # Base score for Pareto points: scale combined to [10, 20] for clear separation
    # Non-Pareto points get 0
    scores = np.where(is_pareto, 10.0 + 10.0 * combined, 0.0)
    
    # Add a small exploration bonus to ensure diversity among Pareto points
    # Points with higher uncertainty_per_cost get slightly higher scores
    scores += 0.1 * upc_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
