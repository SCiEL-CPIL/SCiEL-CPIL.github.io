"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement values."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _hypervolume_contribution(points, ref_point):
    """
    Compute hypervolume contribution of each point in a 2D objective space.
    
    points: (N, 2) array where each row is (cost_norm, -ei_norm)
             We minimize both.
    ref_point: (2,) reference point (worst acceptable values, e.g., [1.0, 0.0])
    
    Returns: (N,) array of hypervolume contributions.
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # Sort by first objective (cost_norm, minimize)
    order = np.argsort(points[:, 0])
    sorted_points = points[order]
    
    contributions = np.zeros(n)
    
    # For each point i, find the "width" in cost space it uniquely dominates
    # and the "height" in -ei_norm space it uniquely dominates.
    
    # We iterate through points sorted by cost.
    # A point i contributes to the hypervolume in the region:
    # cost in [sorted_points[i, 0], next_dominating_cost]
    # -ei_norm in [sorted_points[i, 1], ref_point[1]]
    
    # next_dominating_cost is the cost of the first subsequent point j
    # such that sorted_points[j, 1] <= sorted_points[i, 1] (i.e., j has better or equal -ei_norm)
    # If no such j exists, the cost upper bound is ref_point[0].
    
    # Optimization: We can precompute the minimum -ei_norm from the right.
    # But we need the *first* index where the value drops below the current one.
    # The O(N^2) approach is fine for N=2000.
    
    for i in range(n - 1):
        current_cost = sorted_points[i, 0]
        current_neg_ei = sorted_points[i, 1]
        
        # Find first j > i where sorted_points[j, 1] <= current_neg_ei
        # If found, cost_upper = sorted_points[j, 0]
        # Else, cost_upper = ref_point[0]
        
        # Vectorized search for the first index in the suffix
        suffix_neg_ei = sorted_points[i+1:, 1]
        mask = suffix_neg_ei <= current_neg_ei
        if np.any(mask):
            next_idx = i + 1 + np.argmax(mask)
            cost_upper = sorted_points[next_idx, 0]
        else:
            cost_upper = ref_point[0]
        
        cost_width = cost_upper - current_cost
        if cost_width <= 0:
            contributions[i] = 0.0
            continue
        
        # The height is from current_neg_ei to ref_point[1]
        # Note: ref_point[1] is 0.0 (worst -ei_norm).
        # current_neg_ei is <= 0.
        # So height = 0.0 - current_neg_ei = -current_neg_ei = ei_norm
        ei_height = ref_point[1] - current_neg_ei
        if ei_height <= 0:
            contributions[i] = 0.0
            continue
            
        contributions[i] = cost_width * ei_height
    
    # Last point
    i = n - 1
    current_cost = sorted_points[i, 0]
    current_neg_ei = sorted_points[i, 1]
    
    cost_width = ref_point[0] - current_cost
    ei_height = ref_point[1] - current_neg_ei
    
    if cost_width > 0 and ei_height > 0:
        contributions[i] = cost_width * ei_height
    else:
        contributions[i] = 0.0
        
    # Map back to original order
    result = np.zeros(n)
    result[order] = contributions
    return result


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using normalized Pareto-frontier hypervolume contribution.
    
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
    
    # Compute EI
    ei = _compute_ei(mu, sigma, f_best)
    
    # Check deadline early
    if time.time() - t_start > _DEADLINE * 0.5:
        # Fallback to simple normalized EI/cost
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalization
    # Cost is in [1, 2]. Map to [0, 1].
    cost_norm = (cost - 1.0) / 1.0
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # EI is in [0, max_ei]. Map to [0, 1].
    max_ei = np.max(ei)
    if max_ei < 1e-9:
        # If all EI are 0, scores should be 0 or based on cost?
        # If no improvement is expected, we should pick the cheapest to save budget?
        # Or just return 0.
        # Returning 0 is safe. The loop might stall if all scores are 0.
        # But if max_ei is tiny, it's better to pick the cheapest point to reduce cost.
        # Let's assign score = 1.0 / cost_norm (inverted cost) if EI is negligible.
        # Or just return 0. The evaluator picks argmax. If all 0, it picks index 0.
        # Let's use a small epsilon to allow selection.
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / max_ei
    
    # Build 2D objective space: (cost_norm, -ei_norm)
    # We minimize cost_norm and -ei_norm (which is maximizing ei_norm).
    obj = np.column_stack([cost_norm, -ei_norm])
    
    # Reference point: worst cost_norm (1.0), worst -ei_norm (0.0)
    ref_point = np.array([1.0, 0.0])
    
    # Compute hypervolume contributions
    hv_contrib = _hypervolume_contribution(obj, ref_point)
    
    scores = hv_contrib
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
