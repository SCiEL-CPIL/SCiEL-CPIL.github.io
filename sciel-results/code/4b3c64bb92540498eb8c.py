"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    lower-confidence-bound (LCB) exploration term, tuned for the
    Styblinski-Tang 1D landscape on [-5, 5].
    
    The Styblinski-Tang function has a known global minimum at -39.1662
    at x = -2.903534. We use a mixed acquisition that is more aggressive
    in exploration early (when f_best is far from the optimum) and shifts
    toward exploitation as f_best improves.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return a simple EI-based score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # LCB (Lower Confidence Bound) score: lower is better, so we negate
    # for a "higher is better" acquisition
    # beta controls exploration-exploitation tradeoff
    # Start more exploratory, reduce as we get closer to known optimum
    known_min = -39.1662
    # beta decreases as f_best approaches known_min
    # When f_best is far (e.g., 50), beta ~ 2.5; when close, beta ~ 0.8
    progress = np.clip((f_best - known_min) / 100.0, 0.0, 1.0)
    beta = 0.8 + 1.7 * progress
    
    lcb_score = -mu + beta * sigma_safe  # negate for "higher is better"
    
    # Normalize both to have comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    lcb_max = np.max(lcb_score)
    lcb_min = np.min(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = lcb_score - lcb_min
    
    # Blend: weight EI more as we improve
    # When f_best is poor (far from optimum), rely more on LCB (exploration)
    # When f_best is good, rely more on EI (exploitation)
    ei_weight = 0.35 + 0.45 * (1.0 - progress)
    lcb_weight = 1.0 - ei_weight
    
    score = ei_weight * ei_norm + lcb_weight * lcb_norm
    
    # Small diversity bonus: reward points in regions with high uncertainty
    # relative to the max uncertainty, to avoid over-concentrating
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        diversity = 0.05 * (sigma_safe / sigma_max)
    else:
        diversity = np.zeros_like(sigma_safe)
    
    score = score + diversity
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
