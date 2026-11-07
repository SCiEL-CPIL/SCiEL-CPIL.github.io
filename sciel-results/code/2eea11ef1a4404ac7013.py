"""Rewritten cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with cost normalization.
    Higher score = more likely to be evaluated.
    
    mu, sigma, f_best, cost all have shape (2000,).
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're close to timeout, return a quick fallback
    if time.time() > _DEADLINE:
        scores = -mu / np.maximum(cost, 1e-8)
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is non-negative and handle edge cases
    sigma = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # For minimization, we want to maximize the expected reduction from f_best
    
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF
    # Use scipy if available, otherwise use numpy approximation
    try:
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
    except ImportError:
        # Approximate CDF using error function
        Phi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement for minimization
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Clip EI to be non-negative (should be, but numerical issues can cause negatives)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # This properly accounts for the non-uniform evaluation cost
    score = EI / cost_safe
    
    # Add a small exploration bonus for very uncertain regions
    # This helps when EI is dominated by exploitation but we need to cover space
    # Weight the bonus by how early we are (high f_best means we're still exploring)
    f_best_clipped = np.clip(f_best, 0.0, 100.0)
    exploration_weight = 0.1 * np.exp(-f_best_clipped / 20.0)
    
    # Normalize sigma for the bonus
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Add exploration bonus
    score = score + exploration_weight * sigma_norm / cost_safe
    
    # Safety check for finite values
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
