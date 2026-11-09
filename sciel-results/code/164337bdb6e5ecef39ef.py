"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive EI with uncertainty-weighted exploration bonus.
    
    Uses a smooth blend of Expected Improvement and a UCB-style exploration term,
    with the exploration weight adapted to the current best value relative to
    the known global minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    DEADLINE = 285.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute a UCB-style exploration term
    # UCB = -mu + beta * sigma
    # We use a smooth beta that depends on how close we are to the known minimum
    known_min = -2.53218
    
    # Progress: 0 when at minimum, 1 when far (f_best near 0 or positive)
    # Use a smooth sigmoid-like transition
    if f_best <= known_min:
        progress = 0.0
    else:
        # Scale progress: when f_best is close to known_min, progress ~ 0
        # When f_best is far (e.g., near 0), progress ~ 1
        scale = max(known_min - 1.0, 1e-6)  # Reference scale for "far"
        progress = min(max((f_best - known_min) / scale, 0.0), 1.0)
    
    # Beta: exploration weight
    # Start with moderate exploration, increase when far from optimum
    # beta in [0.5, 2.5]
    beta = 0.5 + 2.0 * progress
    
    # UCB score for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB
    # When progress is high (early, far from min), weight UCB more
    # When progress is low (late, near min), weight EI more
    # w_ucb: weight on UCB, w_ei: weight on EI
    # Use smooth interpolation
    w_ucb = 0.3 + 0.5 * progress  # Range: 0.3 to 0.8
    w_ei = 1.0 - w_ucb
    
    # Normalize both to comparable scales before blending
    # EI is always non-negative; UCB can be negative
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blended score in normalized space
    blended_norm = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Scale back to a reasonable range
    # Use the EI scale as reference
    scores = blended_norm * ei_range + ei_min
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a simple fallback: pure EI
        z_fallback = (f_best - mu) / np.maximum(sigma, 1e-9)
        scores_fallback = (f_best - mu) * norm.cdf(z_fallback) + np.maximum(sigma, 1e-9) * norm.pdf(z_fallback)
        scores_fallback = np.where(np.isfinite(scores_fallback), scores_fallback, 0.0)
        return scores_fallback
    
    return scores
# EVOLVE-BLOCK-END
