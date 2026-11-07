# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# We implement a cost-aware acquisition that blends Expected Improvement (EI)
# with a penalty for expected cost, and adds an exploration term based on
# normalized sigma to encourage diversity early on.
# Additionally, we use a simple deadline guard to avoid timeout.

_start_time = None


def _ensure_start():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _elapsed():
    t0 = _ensure_start()
    return time.time() - t0


def _remaining():
    return 320.0 - _elapsed()  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    scores: array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple score
    if _remaining() < 5.0:
        # Fallback: just use negative cost as a simple score
        return -cost.astype(float)

    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Make EI non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide EI by cost to get improvement per unit cost
    # But also add an exploration bonus based on uncertainty
    # The cost is in [1, 2] range (1 + mean(x_scaled))
    
    # Normalize EI by cost
    cost_safe = np.maximum(cost, 1e-6)
    ei_per_cost = ei / cost_safe
    
    # Exploration term: higher sigma means more uncertainty
    # Normalize sigma to [0, 1] range for the exploration bonus
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_min = np.min(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-8:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Blend exploitation (EI per cost) with exploration (normalized sigma)
    # Use a weight that decreases as we get more certain
    # The weight of exploration should be higher when sigma is large relative to improvement
    # Simple approach: use a fixed blend
    exploration_weight = 0.3
    scores = (1 - exploration_weight) * ei_per_cost + exploration_weight * sigma_norm * (np.max(ei_per_cost) + 1e-8)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
