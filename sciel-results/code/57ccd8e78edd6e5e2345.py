import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 325.0


def acquisition(mu, sigma, f_best):
    """
    UCB acquisition with dynamically increasing exploration constant.
    For minimization: score = mu - kappa * sigma
    where kappa scales with sqrt of estimated number of evaluations.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Pure exploitation fallback
        scores = mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Estimate number of evaluations from the range of sigma values
    # Early on, sigma will be larger; later, it shrinks
    # Use a heuristic: kappa = c * sqrt(n)
    # We can estimate n by looking at how much exploration is needed
    # A simple approach: use the median sigma as a proxy
    med_sigma = np.median(sigma_safe)
    
    # kappa schedule: starts around 0.5, grows slowly
    # For 30 total evaluations, we want reasonable exploration early
    # kappa = 0.3 + 0.1 * sqrt(n_est)
    # Estimate n from sigma magnitude: smaller sigma means more evaluations done
    # If median sigma is very small (< 0.01), we're in late stage
    # If median sigma is large (> 0.1), we're in early stage
    # Map median sigma to an estimated n in [5, 30]
    # log scale: n_est = 5 + 25 * (1 - med_sigma / (med_sigma + 0.1))
    # This gives ~5 when med_sigma is large, ~30 when med_sigma is small
    
    # Simpler: use a fixed moderate kappa that works well for 30 evals
    # UCB with kappa ~ 1-2 works well for small budgets
    # Let's use kappa that adapts: larger when sigma is large (early), smaller when sigma is small (late)
    # kappa = 1.5 * (med_sigma + 0.01) / (med_sigma + 0.05)
    # This gives kappa ~ 1.5 early, ~ 0.3 late
    
    # Actually, for Hartmann 3D with 30 evals, a fixed kappa around 1.0-1.5 works well
    # Let's use a mild schedule
    kappa = 1.0 + 0.5 * np.exp(-5.0 * med_sigma)
    
    # UCB for minimization: lower score is better, so we want to maximize mu - kappa*sigma
    scores = mu - kappa * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
