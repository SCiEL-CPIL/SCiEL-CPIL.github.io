"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level start time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition function for Bayesian optimization.
    Combines adaptive LCB (exploration) with Expected Improvement (exploitation),
    weighted by a sigmoid schedule based on f_best, and normalized by evaluation cost.
    
    Score = w * (LCB_improvement / cost) + (1-w) * (EI / cost)
    """
    # Ensure inputs are 1D arrays for consistent shape handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)
    
    # Clamp cost for numerical stability
    cost_safe = np.maximum(cost, 1e-10)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-6)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Deadline check (unlikely to be triggered given O(N) vectorized ops)
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: simple cost-aware LCB
        simple_scores = (1.5 * sigma_safe - mu) / cost_safe
        return np.where(np.isfinite(simple_scores), simple_scores, -1e9)
    
    # Adaptive beta schedule:
    # - Uses log(1 + f_best) for smooth decay.
    # - When f_best is large (early), beta is large -> high exploration (sigma term dominates).
    # - When f_best is small (late, near optimum), beta -> 1.0 -> exploitation (mu term dominates).
    f_best_clamped = max(float(f_best), 0.0)
    beta = 1.0 + 2.0 * np.log1p(f_best_clamped)
    
    # LCB Improvement: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    lcb_improvement = f_best_clamped - mu + beta * sigma_safe
    lcb_improvement = np.maximum(lcb_improvement, 0.0)
    
    # Expected Improvement (EI) for minimization
    z = (f_best_clamped - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    ei = (f_best_clamped - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Cost-weighted scores
    lcb_score = lcb_improvement / cost_safe
    ei_score = ei / cost_safe
    
    # Stage-Adaptive Weighting:
    # Sigmoid: if f_best >> T, w -> 1 (High LCB/exploration). If f_best << T, w -> 0 (High EI/exploitation).
    T = 0.5  # Threshold for transition
    K = 8.0  # Steepness of transition
    arg = K * (f_best_clamped - T)
    arg = np.clip(arg, -50, 50)
    w_lcb = 1.0 / (1.0 + np.exp(-arg))
    
    # Combine scores
    scores = w_lcb * lcb_score + (1.0 - w_lcb) * ei_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
