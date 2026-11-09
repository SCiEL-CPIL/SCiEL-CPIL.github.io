"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines EI (exploitation) with a UCB-like term (exploration).
    For minimization: we want low mu (good predictions) and high sigma (uncertainty).
    
    The key insight from the cell findings: linear optimistic scoring (UCB-like) provides
    stronger exploration pressure than probability-saturated EI. We use a hybrid approach
    that weights both, but with emphasis on the UCB component for better exploration.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability: clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 1e-12)
    
    # UCB-like term for exploration: -mu + beta * sigma
    # For minimization, we want low mu (attractive) and high sigma (uncertainty bonus)
    beta = 3.0  # Strong exploration weight based on cell findings (1.538 score)
    ucb = -mu + beta * sigma_safe
    
    # Normalize both terms to comparable scales using robust statistics
    ei_median = np.median(ei)
    ei_mad = np.median(np.abs(ei - ei_median))
    if ei_mad < 1e-10:
        ei_mad = 1.0
    
    ucb_median = np.median(ucb)
    ucb_mad = np.median(np.abs(ucb - ucb_median))
    if ucb_mad < 1e-10:
        ucb_mad = 1.0
    
    ei_norm = (ei - ei_median) / (ei_mad + 1e-10)
    ucb_norm = (ucb - ucb_median) / (ucb_mad + 1e-10)
    
    # Clip to prevent extreme outliers from dominating
    ei_norm = np.clip(ei_norm, -10, 10)
    ucb_norm = np.clip(ucb_norm, -10, 10)
    
    # Combine with weights favoring UCB for stronger exploration
    w_ei = 0.4
    w_ucb = 0.6
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're running low on time, return a simpler score
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback to pure UCB which is faster to compute
        return ucb
    
    return scores
# EVOLVE-BLOCK-END
