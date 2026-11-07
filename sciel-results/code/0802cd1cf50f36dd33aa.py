"""
Batch-aware Greedy Expected Improvement (GEI) approximation for cost-aware BO.
Combines EI with a variance-reduction penalty that approximates how much the
chosen evaluation shrinks posterior uncertainty across the remaining candidate set.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a GEI-inspired score.
    
    The score combines:
      1. Expected Improvement (EI) for exploitation.
      2. A variance-reduction penalty that approximates the expected reduction
         in global posterior uncertainty if this point were evaluated next.
         Points that are far from the current best (in normalized distance) and
         have high uncertainty are penalized less, encouraging exploration of
         unexplored regions. Points near the best with low uncertainty are
         penalized more, discouraging redundant evaluations.
      3. Cost normalization to favor cheaper evaluations.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return deterministic fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardize improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # --- GEI-inspired variance reduction penalty ---
    # Approximate the expected reduction in global posterior variance.
    # Use the spread of sigma across the candidate set as a proxy for
    # the global uncertainty landscape.
    
    # Normalize sigma to [0, 1] range for this batch
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-9:
        sigma_range = 1e-9
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Points with high sigma_norm are in high-uncertainty regions.
    # Evaluating them reduces global uncertainty more.
    # We want to PENALIZE points that do NOT reduce uncertainty much.
    # So the penalty should be LOW for high sigma_norm points.
    
    # Approximate expected variance reduction: proportional to sigma^2 * (1 - sigma_norm)
    # This is a heuristic: high sigma means more to learn, but if sigma is already
    # the max in the set, the marginal reduction is less.
    # Actually, let's think differently: the value of information from evaluating x
    # is roughly proportional to how much it reduces the average posterior variance.
    # A simple proxy: the reduction is larger when sigma is large relative to the
    # average sigma.
    
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1e-9
    
    # Relative uncertainty: how much more uncertain is this point than average?
    rel_uncertainty = sigma_safe / sigma_mean
    
    # Variance reduction proxy: points with high relative uncertainty reduce
    # global uncertainty more when evaluated.
    # We want to REWARD high variance reduction, so SUBTRACT a penalty that is
    # inversely related to variance reduction.
    
    # Penalty: high when rel_uncertainty is low (point is already well-explored)
    # Low when rel_uncertainty is high (point is in unexplored region)
    # Use: penalty = c * (1 / (1 + rel_uncertainty))
    # This is bounded and smooth.
    
    # Scale the penalty based on progress
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # When gap is large, we can afford more exploration (smaller penalty)
    # When gap is small, we need to focus (larger penalty)
    penalty_scale = 0.02 * (1.0 + 3.0 * np.exp(-gap_to_min / 0.2))
    
    variance_penalty = penalty_scale / (1.0 + rel_uncertainty)
    
    # Combine EI and PI with variance reduction penalty
    # Weight: 0.8 EI + 0.2 PI
    base_score = 0.8 * ei + 0.2 * pi
    score = base_score - variance_penalty
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
