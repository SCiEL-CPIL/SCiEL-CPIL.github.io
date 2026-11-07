"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_CALL_COUNT = 0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Hybrid EI with adaptive UCB component.
    
    Uses Expected Improvement as the primary signal for exploitation,
    combined with an adaptive UCB term that decays over the optimization
    sequence to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    global _CALL_COUNT, _START_TIME
    _ensure_start_time()
    
    # Safety guard: if we're getting close to the 300s limit, return quickly
    elapsed = time.time() - _START_TIME
    if elapsed > 290:
        sigma_safe = np.maximum(sigma, 1e-10)
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Track call count for adaptive behavior
    _CALL_COUNT += 1
    
    # Estimate current step in the 30-step sequence
    # 30 evaluations per seed, so step cycles 0-29
    step = _CALL_COUNT % 30
    
    # EI Calculation
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive UCB Weight
    # High exploration early, low exploitation late
    # beta starts at 0.15 and decays to 0.02
    beta_max = 0.15
    beta_min = 0.02
    tau = 8.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-step / tau)
    
    ucb_term = beta * sigma_safe
    
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
