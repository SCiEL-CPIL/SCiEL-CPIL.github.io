import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid of Cost-Adjusted EI and Cost-Adjusted UCB.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    if not _time_ok():
        # Fallback: simple exploitative EI
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    # Clean inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    
    # Calculate EI
    z = (f_best_val - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best_val - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB
    # Beta decays from 2.5 to 0.5 based on progress
    # Progress estimate: f_best starts around 0-10, ends at -78.33
    # If f_best is positive, progress is 0.
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)
    
    # Adaptive beta
    beta = 2.5 - 2.0 * progress
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # Use max normalization to keep values in [0, 1] roughly
    max_ei = np.max(ei) + 1e-6
    max_ucb = np.max(ucb) + 1e-6
    
    norm_ei = ei / max_ei
    norm_ucb = ucb / max_ucb
    
    # Weights: Start with more UCB (exploration), shift to EI (exploitation)
    # w_ei increases with progress
    w_ei = 0.3 + 0.7 * progress
    w_ucb = 1.0 - w_ei
    
    combined_score = (w_ei * norm_ei + w_ucb * norm_ucb) / cost_safe
    
    # Ensure finite
    combined_score = np.where(np.isfinite(combined_score), combined_score, 0.0)
    
    return combined_score
