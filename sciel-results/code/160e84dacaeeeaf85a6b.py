"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time since module load for deadline guard
_MODULE_START = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 20.0


def _time_remaining():
    elapsed = time.time() - _MODULE_START
    return _DEADLINE - elapsed - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best):
    """
    Quantile-binned LCB acquisition for systematic spatial coverage.
    
    Strategy:
    - Partition the 2000 candidates into disjoint quantile bins of the posterior mean.
    - Within each bin, compute LCB = mu - beta * sigma (lower confidence bound for minimization).
    - Score = -LCB normalized within each bin, so the best candidate per bin gets highest score.
    - This forces one evaluation per spatial region while still preferring low expected values.
    
    mu, sigma: shape (2000,)
    f_best: scalar, lowest observed value
    Returns: shape (2000,) scores, higher = evaluate next
    """
    # Time guard: if we're running out of time, return a simple EI score
    if _time_remaining() < 0.5:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(n)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute LCB for minimization: lower confidence bound
    # beta controls exploration-exploitation tradeoff
    # Use a moderate beta that adapts to the scale of sigma
    beta = 1.5
    
    # LCB = mu - beta * sigma (lower bound on the function value)
    lcb = mu - beta * sigma_safe
    
    # Partition into quantile bins of posterior mean for systematic coverage
    order = np.argsort(mu)
    n_bins = 20
    bin_size = n // n_bins
    
    scores = np.zeros(n)
    
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < n_bins - 1 else n
        bin_indices = order[bin_start:bin_end]
        
        # Within this bin, score based on -LCB (we want low LCB = high score)
        bin_lcb = lcb[bin_indices]
        
        # Normalize -LCB within the bin to [0, 1]
        neg_lcb = -bin_lcb
        neg_lcb_min = np.min(neg_lcb)
        neg_lcb_max = np.max(neg_lcb)
        if neg_lcb_max - neg_lcb_min > 1e-12:
            bin_scores = (neg_lcb - neg_lcb_min) / (neg_lcb_max - neg_lcb_min)
        else:
            bin_scores = np.ones_like(bin_indices)
        
        scores[bin_indices] = bin_scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
