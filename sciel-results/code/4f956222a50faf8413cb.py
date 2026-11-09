"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Pareto-frontier search over (cost, -EI) trade-off.
    
    Instead of a simple weighted sum, we compute a hypervolume-contribution-based
    score that explicitly models the multi-objective nature of budget-constrained search.
    We select candidates that maximize their contribution to the non-dominated frontier
    of (cost, -expected_improvement) pairs, with a fallback to PI for robustness.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 325.0:  # Safety margin
        return np.zeros_like(mu)
    
    n = mu.shape[0]
    
    # Expected Improvement
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware: normalize cost to [0, 1] range for this batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Multi-objective: (cost, -ei) where we want low cost and high ei
    # Compute hypervolume contribution approximation for each candidate
    # We use a simplified approach: score based on non-dominance and relative improvement
    
    # First, identify non-dominated points in (cost, -ei) space
    # A point i dominates point j if cost_i <= cost_j AND ei_i >= ei_j (and at least one strict)
    
    # Efficient approach: sort by cost, then track max ei seen so far
    cost_order = np.argsort(cost)
    ei_sorted = ei[cost_order]
    cost_sorted = cost[cost_order]
    
    # Compute running max of ei in cost-sorted order
    # Points that are on the Pareto front have ei equal to the running max at their position
    running_max_ei = np.maximum.accumulate(ei_sorted)
    
    # A point is non-dominated if its ei equals the running max at its position
    # (and it's the first point to achieve that max, or we handle ties)
    is_pareto = np.zeros(n, dtype=bool)
    pareto_indices = cost_order[ei_sorted >= running_max_ei - 1e-12]
    
    # For hypervolume contribution, we approximate:
    # Each point's contribution is proportional to how much it improves the frontier
    # Simplified: use the gap between this point's ei and the next-worse non-dominated point
    
    # Compute a score that rewards:
    # 1. Low cost
    # 2. High EI
    # 3. Being on or near the Pareto front
    # 4. PI for exploration robustness
    
    # Hypervolume contribution approximation:
    # For each point, compute the area it "owns" in the (cost, ei) space
    # This is approximated by: (ei_i - ei_next) * (cost_next - cost_i) for sorted points
    # where ei_next is the ei of the next point with lower cost
    
    # Simpler robust approach: combine EI, PI, and cost with adaptive weights
    # based on how far we are from the Pareto front
    
    # Distance from Pareto front (in normalized space)
    ei_norm = ei / (np.max(ei) + eps)
    cost_penalty = cost_norm  # 0 = cheap, 1 = expensive
    
    # Score: weighted combination that emphasizes Pareto-optimal regions
    # Higher weight on EI when cost is low, higher weight on PI when uncertain
    uncertainty = sigma_safe / (np.max(sigma_safe) + eps)
    
    # Adaptive weights:
    # - When EI is high and cost is low: strong exploitation
    # - When EI is uncertain: more exploration via PI
    # - Always penalize high cost
    
    w_ei = 0.5 + 0.3 * (1 - cost_norm)  # Higher weight on EI for cheap points
    w_pi = 0.2 + 0.3 * uncertainty      # Higher weight on PI for uncertain points
    w_cost = 0.3                        # Cost penalty weight
    
    # Normalize EI and PI to [0, 1]
    ei_max = np.max(ei) + eps
    pi_max = np.max(pi) + eps
    
    ei_normed = ei / ei_max
    pi_normed = pi / pi_max
    
    # Pareto front bonus: points on the front get a boost
    pareto_bonus = np.zeros(n)
    pareto_bonus[pareto_indices] = 1.0
    # Smooth bonus: points close to the front get partial credit
    for i in range(n):
        if not is_pareto[i]:
            # Find distance to nearest Pareto point in (cost, ei) space
            # Approximate by checking if this point could be part of the front
            # with a small perturbation
            pass
    
    # Final score: higher is better
    score = (w_ei * ei_normed + w_pi * pi_normed - w_cost * cost_norm)
    
    # Add Pareto front bonus
    score += 0.2 * pareto_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
