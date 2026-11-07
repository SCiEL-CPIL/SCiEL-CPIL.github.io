"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Hybrid Expected Improvement (EI) with a tuned UCB exploration bonus.
    
    This approach combines the exploitation strength of Expected Improvement, which is 
    particularly effective for small evaluation budgets, with a mild exploration term 
    (proportional to posterior standard deviation) to help navigate the multimodal 
    landscape of Hartmann-3D.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Safety guard: if we're getting close to the 300s limit, return a fast fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-10)
        # Fallback: Simple UCB-like score
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative and safe for operations
    # Using a small epsilon to avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in norm.cdf/pdf
    # Values beyond +/- 10 are effectively 0 or 1 for CDF and ~0 for PDF
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If mu < f_best, z > 0, Phi(z) is high, EI is positive.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an exploration bonus (UCB-like component)
    # This term encourages sampling in areas of high uncertainty.
    # Increased weight from 0.01 to 0.05 to better handle the multimodal landscape
    # in the early stages of optimization (small N).
    exploration_weight = 0.05
    ucb_component = exploration_weight * sigma_safe
    
    scores = ei + ucb_component
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
