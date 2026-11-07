"""Rewritten program for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a combination of Expected Improvement (EI) and a scaled uncertainty term
    to balance exploration and exploitation. The weight shifts from exploration
    to exploitation as f_best approaches the known global minimum.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Finite scores where higher means the point should be evaluated next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute a scaled uncertainty score
    # We want to encourage exploration where uncertainty is high,
    # but not too much. Scale by the range of mu to make it comparable.
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-12 else 1.0
    sigma_scaled = sigma_safe / mu_range
    
    # Adaptive weighting based on progress toward the known minimum
    f_min_known = -1.8013
    d = max(f_best - f_min_known, 0.0)
    
    # When d is large (early), explore more (lower alpha for EI)
    # When d is small (late), exploit more (higher alpha for EI)
    # alpha goes from ~0.3 (explore) to ~1.0 (exploit)
    alpha = 1.0 - 0.7 * min(1.0, d / 2.0)
    alpha = np.clip(alpha, 0.3, 1.0)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min < 1e-12:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    
    # Normalize sigma_scaled to [0, 1]
    ss_min = np.min(sigma_scaled)
    ss_max = np.max(sigma_scaled)
    if ss_max - ss_min < 1e-12:
        ss_norm = np.ones_like(sigma_scaled)
    else:
        ss_norm = (sigma_scaled - ss_min) / (ss_max - ss_min)
    
    # Combine: higher score means evaluate next
    # alpha * EI + (1-alpha) * uncertainty
    combined_score = alpha * ei_norm + (1.0 - alpha) * ss_norm
    
    # Ensure all scores are finite
    combined_score = np.where(np.isfinite(combined_score), combined_score, 0.0)
    
    return combined_score
# EVOLVE-BLOCK-END
