"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with adaptive weighting based on uncertainty.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Lower Confidence Bound (beta controls exploration)
    # Adaptive beta: higher when uncertainty is high relative to improvement
    # Use a fixed beta that balances exploration and exploitation
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if lcb_max > lcb_min:
        lcb_norm = (lcb - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # Adaptive weighting: weight LCB more when sigma is high (uncertain regions)
    # and weight EI more when we have good improvements
    # Use a smooth transition based on relative uncertainty
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 0:
        # Where sigma is high relative to mean, prefer exploration (LCB)
        # Where sigma is low, prefer exploitation (EI)
        uncertainty_ratio = sigma_safe / (sigma_mean + 1e-10)
        # Smooth weight: higher weight for LCB in uncertain regions
        w_lcb = 0.3 + 0.4 * np.clip(uncertainty_ratio, 0, 2) / 2.0
        w_ei = 1.0 - w_lcb
        # Clip weights to reasonable range
        w_lcb = np.clip(w_lcb, 0.2, 0.7)
        w_ei = 1.0 - w_lcb
    else:
        w_lcb = np.full_like(ei, 0.5)
        w_ei = np.full_like(ei, 0.5)
    
    # Combine
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
