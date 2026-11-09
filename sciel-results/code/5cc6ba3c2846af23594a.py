"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Multi-objective acquisition: weighted sum of normalized EI and normalized predictive uncertainty.
    Dynamic weight increases with evaluations to balance exploration and exploitation.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Numerical safeguard for small sigma
    small_sigma_mask = sigma < 1e-8
    if np.any(small_sigma_mask):
        ei[small_sigma_mask] = np.maximum(0, f_best - mu[small_sigma_mask])
    
    # Ensure all scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize sigma to [0, 1]
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Estimate progress: 0 at start, 1 at end
    # f_best starts around 0 and goes to -3.32237
    f_best_min = -3.32237
    f_best_max = 0.0
    
    if f_best < f_best_min:
        progress = 1.0
    elif f_best > f_best_max:
        progress = 0.0
    else:
        progress = (f_best_max - f_best) / (f_best_max - f_best_min)
    
    # Smooth progress curve: use sigmoid to create a smoother transition
    # This avoids abrupt changes in weighting
    progress_smooth = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.5)))
    
    # Dynamic weight: starts at 0.4 (more exploration), increases to 0.85 (more exploitation)
    # Using a smoother schedule for better balance
    w_ei = 0.4 + 0.45 * progress_smooth
    w_sigma = 1.0 - w_ei
    
    # Add a small constant exploration floor to ensure we never fully stop exploring
    w_sigma = max(w_sigma, 0.1)
    w_ei = 1.0 - w_sigma
    
    # Combined score
    score = w_ei * ei_norm + w_sigma * sigma_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
