"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement with uncertainty-aware xi parameter.
    
    The key structural idea: use standard EI but with xi that adapts to the
    mean posterior standard deviation. When uncertainty is high (early in the
    search or in poorly-explored regions), xi is larger, encouraging exploration
    of regions that might contain values significantly better than f_best.
    When uncertainty is low (later in the search), xi shrinks toward zero,
    focusing on exploitation of the current best.
    
    This avoids the pitfalls of fixed-parameter EI (which can get stuck in
    local minima) and the randomness of Thompson sampling (which can be
    inconsistent across runs).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Pure exploitation fallback
        return -mu
    
    n = len(mu)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive xi: scales with the mean posterior uncertainty
    # When sigma_mean is large relative to the spread of mu, we're in early
    # exploration phase -> larger xi
    # When sigma_mean is small, we're in exploitation phase -> smaller xi
    
    sigma_mean = np.mean(sigma_safe)
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Normalize sigma_mean by the typical scale of the problem
    # The ratio sigma_mean / mu_range tells us how much uncertainty remains
    ratio = sigma_mean / mu_range
    
    # Map ratio to xi: 
    # - Very high uncertainty (ratio >> 1): xi ~ 0.5 * sigma_mean (strong exploration)
    # - Moderate uncertainty: xi ~ 0.1 * sigma_mean
    # - Low uncertainty (ratio << 1): xi ~ 0.01 * sigma_mean (mild exploration)
    # Use a smooth, bounded mapping
    xi_scale = 0.05 + 0.45 * min(ratio, 2.0) / 2.0
    xi = xi_scale * sigma_mean
    
    # Clip xi to avoid pathological values
    xi = np.clip(xi, 0.0, 0.5 * sigma_mean)
    
    # Compute EI with adaptive xi
    # EI(x) = (f_best - xi - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - xi - mu(x)) / sigma(x)
    
    z = (f_best - xi - mu) / sigma_safe
    
    # Use stable computation of EI
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For points where f_best - xi - mu is very negative, EI should be ~0
    # norm.cdf handles this naturally, but let's ensure numerical stability
    ei = np.maximum(ei, 0.0)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
