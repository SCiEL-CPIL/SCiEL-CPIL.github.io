"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 10.0
_TOTAL_BUDGET = 330.0


def _time_remaining():
    return _TOTAL_BUDGET - _DEADLINE_MARGIN - (time.time() - _start_time)


def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling + Expected Improvement acquisition function.
    
    Combines Thompson Sampling (structural randomness for exploration) with
    Expected Improvement (principled exploitation) via an adaptive blend.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if time is running out, return pure exploitation
    if _time_remaining() < 5.0:
        return -mu.copy()
    
    n = len(mu)
    
    # Compute Expected Improvement for minimization
    safe_sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    z_ei = (f_best - mu) / safe_sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + safe_sigma * norm.pdf(z_ei)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Thompson Sampling: draw one sample from the posterior
    z_ts = np.random.randn(n)
    ts_sample = mu + sigma * z_ts
    ts_score = -ts_sample  # higher = lower sampled value = more promising
    
    # Adaptive blend: weight shifts from exploration to exploitation
    # progress: 1 when f_best is large (early), 0 when f_best is small (late)
    progress = f_best / (f_best + 1.0)
    
    # Early: more Thompson Sampling (exploration via randomness)
    # Late: more EI (exploitation via principled improvement)
    # ts_weight goes from ~0.7 (early) to ~0.2 (late)
    ts_weight = 0.2 + 0.5 * progress
    ei_weight = 1.0 - ts_weight
    
    # Normalize both scores to comparable scale before blending
    # Use robust normalization: center by median, scale by IQR
    def _robust_normalize(arr):
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-10:
            return arr - med
        return (arr - med) / iqr
    
    ei_norm = _robust_normalize(ei)
    ts_norm = _robust_normalize(ts_score)
    
    # Blend
    scores = ts_weight * ts_norm + ei_weight * ei_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
