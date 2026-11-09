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
    
    points: (N, 2) array where each row is (cost, -EI)
             We minimize cost and maximize EI, so we minimize (cost, -EI).
    ref_point: (2,) reference point (worst acceptable values)
    
    Returns: (N,) array of hypervolume contributions.
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # Sort by cost (first objective, minimize)
    order = np.argsort(points[:, 0])
    sorted_points = points[order]
    
    # The reference point defines the "worst" corner
    # For minimization of (cost, -EI), the hypervolume is the area dominated
    # by the point but not by any other point, relative to ref_point.
    
    # A point dominates another if it's better in all objectives.
    # Hypervolume contribution of point i = area of rectangle it uniquely dominates.
    
    # For 2D: sort by first obj. For each point, find the next point that
    # has a smaller second objective. The contribution is the width * height
    # of the unique region.
    
    contributions = np.zeros(n)
    
    # Work with sorted points
    # For point i in sorted order, its unique contribution comes from the
    # region where cost is in [points[i,0], next_dominating_cost] and
    # -EI is in [points[i,1], ref_point[1]]
    
    # Find for each point the next point (in cost order) that has lower -EI
    # (i.e., higher EI), which would dominate in the second objective.
    
    min_neg_ei = np.minimum.accumulate(sorted_points[:, 1])[::-1]
    # min_neg_ei[i] = minimum -EI from i to end
    
    for i in range(n - 1):
        # The cost range this point uniquely covers:
        # From sorted_points[i, 0] to the cost of the first subsequent point
        # that has -EI <= sorted_points[i, 1]
        
        # Find first j > i where sorted_points[j, 1] <= sorted_points[i, 1]
        # If no such j, use ref_point[0] as upper bound
        mask = sorted_points[i+1:, 1] <= sorted_points[i, 1]
        if np.any(mask):
            next_idx = i + 1 + np.argmax(mask)
            cost_upper = sorted_points[next_idx, 0]
        else:
            cost_upper = ref_point[0]
        
        cost_width = cost_upper - sorted_points[i, 0]
        if cost_width <= 0:
            contributions[i] = 0
            continue
        
        # The -EI range: from sorted_points[i, 1] to ref_point[1]
        # But capped by the minimum -EI of subsequent points
        ei_height = ref_point[1] - sorted_points[i, 1]
        if ei_height <= 0:
            contributions[i] = 0
            continue
        
        contributions[i] = cost_width * ei_height
    
    # Last point
    i = n - 1
    cost_width = ref_point[0] - sorted_points[i, 0]
    ei_height = ref_point[1] - sorted_points[i, 1]
    if cost_width > 0 and ei_height > 0:
        contributions[i] = cost_width * ei_height
    
    # Map back to original order
    result = np.zeros(n)
    result[order] = contributions
    return result


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
    
    # Compute EI
    ei = _compute_ei(mu, sigma, f_best)
    
    # Check deadline early
    if time.time() - t_start > _DEADLINE * 0.5:
        # Fallback to simple EI/cost
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Build 2D objective space: (cost, -EI) - we minimize both
    # Points with zero EI have no value; give them minimal score
    obj = np.column_stack([cost, -ei])
    
    # Reference point: worst acceptable cost and worst acceptable -EI
    # Cost range is [1, 2], so ref cost = max(cost) + margin
    # -EI range: worst is 0 (no improvement), so ref -EI = 0 + small
    # Actually, for hypervolume we need a reference point that is worse than all points
    ref_cost = np.max(cost) + 0.1
    ref_neg_ei = 0.01  # slightly positive since -EI <= 0, we want ref to be worse (larger)
    # Wait: we're minimizing -EI. The worst -EI is the largest (closest to 0 or positive).
    # Since EI >= 0, -EI <= 0. The "worst" (largest) -EI is 0.
    # So ref_point should be (ref_cost, 0) or slightly above.
    ref_point = np.array([ref_cost, 0.0])
    
    # For points with -EI >= 0 (i.e., EI <= 0), they are at or beyond the reference
    # in the second objective, so they contribute 0 hypervolume.
    # Only points with -EI < 0 (EI > 0) can contribute.
    
    # Compute hypervolume contributions
    hv_contrib = _hypervolume_contribution(obj, ref_point)
    
    # The hypervolume contribution already captures the trade-off:
    # A point gets credit for the unique region of (cost, -EI) space it dominates.
    # Points that are cheap AND have good EI get large contributions.
    # Points that are expensive but have slightly better EI get small contributions
    # if the cost increase isn't justified.
    
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
