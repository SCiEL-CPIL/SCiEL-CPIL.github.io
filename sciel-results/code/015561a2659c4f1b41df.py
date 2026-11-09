# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for time tracking
_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def _get_elapsed():
    """Return elapsed time since module load, initializing on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded():
    """Check if we are close to the 330s wall-clock limit."""
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Adaptive EI+UCB blend with time-based exploration-to-exploitation shift.
    
    In early iterations (when f_best is still poor), we weight UCB (exploration)
    more heavily. As f_best improves toward the minimum (0), we shift toward
    EI (exploitation). This provides better balance for finding the global
    minimum of the Weierstrass function.
    
    Score = w_ei * EI + w_ucb * UCB
    where w_ei + w_ucb = 1, and the weights depend on f_best.
    """
    # Deadline guard: if we're close to the limit, use a fast fallback
    if _is_deadline_exceeded():
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
        if not np.any(np.isfinite(ei_scores)):
            return np.zeros_like(mu)
        return ei_scores
    
    # Ensure sigma is positive and avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Upper Confidence Bound (for minimization: -mu + beta * sigma)
    # We use an adaptive beta based on f_best
    # When f_best is high (poor), beta is large (more exploration)
    # When f_best is low (good, close to 0), beta is small (more exploitation)
    
    # Normalize f_best: Weierstrass min is 0, and initial values are typically O(1)
    # Use a sigmoid-like transition based on f_best
    # f_best ranges from ~1 (initial) to ~0 (optimal)
    # Let's use: beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    beta_min = 0.1
    beta_max = 2.0
    scale = 0.5
    beta = beta_min + (beta_max - beta_min) * np.exp(-max(f_best, 0) / scale)
    
    # UCB score: -mu + beta * sigma (higher is better for exploration)
    ucb = -mu + beta * sigma_safe
    
    # Adaptive blend: early on (high f_best), weight UCB more
    # Later (low f_best), weight EI more
    # Use a smooth transition based on f_best
    # w_ucb = exp(-f_best / scale2), w_ei = 1 - w_ucb
    scale2 = 0.3
    w_ucb = np.exp(-max(f_best, 0) / scale2)
    w_ei = 1.0 - w_ucb
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.any(np.isfinite(ei)) else 1.0
    ucb_max = np.max(ucb) if np.any(np.isfinite(ucb)) else 1.0
    
    # Avoid division by zero
    ei_norm = ei / (ei_max + eps)
    ucb_norm = ucb / (ucb_max + eps)
    
    # Combined score
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all are -inf, return zeros to avoid invalidity
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
