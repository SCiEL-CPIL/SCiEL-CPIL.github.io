# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_check(safety_margin=5.0):
    """Return True if we are safe to continue."""
    return _get_elapsed() < 330.0 - safety_margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    Uses a combination of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    to balance exploration and exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (Lower Confidence Bound) - for minimization
    # LCB = mu - beta * sigma
    # We want to minimize, so lower LCB is better
    # But we return scores where higher is better
    # So we return -LCB = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # Combine EI and LCB
    # Normalize both to comparable scales
    # EI is already in objective units
    # LCB score is also in objective units (but negated)
    
    # Weighted combination
    w_ei = 0.7
    w_lcb = 0.3
    
    # Normalize LCB score to be comparable with EI
    # Scale LCB by a factor related to the typical magnitude of EI
    # A simple approach: use the mean EI as a scale reference
    ei_mean = np.mean(ei)
    if ei_mean > 1e-9:
        lcb_normalized = lcb_score / ei_mean
    else:
        lcb_normalized = lcb_score
    
    score = w_ei * ei + w_lcb * lcb_normalized
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
