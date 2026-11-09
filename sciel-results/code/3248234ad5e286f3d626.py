"""Cost-aware acquisition with greedy batch diversity: EI/UCB blend with spatial redundancy penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard (shared across calls in the loop)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s wall-clock limit per evaluation; use safety margin

def _elapsed_ok():
    """Return True if we have at least ~5s margin before the hard limit."""
    return (time.time() < _DEADLINE - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization with greedy batch diversity.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple score quickly
    if not _elapsed_ok():
        sigma_safe = np.maximum(sigma, 1e-6)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = len(mu)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Robust sigma: ensure it's positive and not too small
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    ei_cost = ei / cost_safe
    
    # Normalize EI/cost to [0, 1] for combination
    ei_c_min = np.min(ei_cost)
    ei_c_max = np.max(ei_cost)
    ei_c_range = ei_c_max - ei_c_min
    if ei_c_range > 1e-12:
        ei_c_norm = (ei_cost - ei_c_min) / ei_c_range
    else:
        ei_c_norm = np.ones_like(ei_cost)
    
    # UCB: -mu + kappa * sigma (negative because we minimize)
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Base score: weighted combination
    base_score = 0.7 * ei_c_norm + 0.3 * ucb_norm
    base_score = np.where(np.isfinite(base_score), base_score, 0.0)
    
    # Greedy batch selection to encourage diversity
    # We simulate selecting a small batch and use that to create a diversity-aware score
    # Since we only select 1 point per iteration in the actual loop, we use a proxy:
    # penalize candidates that are "close" in score-space to the top candidate
    
    # Find the top candidate index
    top_idx = np.argmax(base_score)
    
    # Compute a diversity penalty based on similarity to the top candidate
    # Use sigma as a proxy for spatial proximity (high sigma = uncertain = potentially far from data)
    # and mu difference as another indicator
    # Simple approach: penalize points that have similar mu and sigma to the top point
    # This is a heuristic since we don't have coordinates
    
    # Alternative: use the rank-based approach
    # Sort by base_score and assign diversity bonuses to top candidates
    # This encourages the loop to explore different regions
    
    # Create a diversity score: higher for candidates that are "different" from the current best
    # Use a simple heuristic: candidates with high uncertainty (high sigma) get a boost
    # since they represent unexplored regions
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / n  # [0, 1], higher = more uncertain
    
    # Combine: base score with a small diversity boost for high-uncertainty points
    diversity_weight = 0.1
    score = (1 - diversity_weight) * base_score + diversity_weight * sigma_rank
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
