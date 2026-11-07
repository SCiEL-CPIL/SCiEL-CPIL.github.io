"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Track time to ensure we don't exceed the 300s limit (with safety margin)
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive weights and beta, driven by the current posterior
    uncertainty (mean sigma).
    
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
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the time limit
    elapsed = time.time() - _start_time
    if elapsed > 290:  # 10s safety margin before 300s limit
        # Return a simple score based on mean to be safe
        return -mu
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Compute UCB with Adaptive Beta
    # Beta should be higher when uncertainty is high.
    # Use mean sigma as a proxy for the stage of optimization.
    mean_sigma = np.mean(sigma_safe)
    
    # Normalize mean_sigma to [0, 1] range for weight calculation.
    # Goldstein-Price values range roughly 3 to 40.
    # Initial sigma might be large (e.g., 10-20), final sigma small (<1).
    # We assume a scale where sigma > 10 is "very early" and sigma < 1 is "late".
    # Let's map mean_sigma to a factor in [0, 1].
    # Using a sigmoid-like mapping or simple clipping.
    sigma_scale = 10.0 
    exploration_factor = np.clip(mean_sigma / sigma_scale, 0.0, 1.0)
    
    # Adaptive Beta: Range from 0.5 (exploit) to 2.5 (explore)
    beta = 0.5 + 2.0 * exploration_factor
    
    # UCB Score: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # 3. Normalize both EI and UCB to [0, 1] for stable blending
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # 4. Adaptive Weights
    # Early on (high exploration_factor): weight UCB more.
    # Late on (low exploration_factor): weight EI more.
    # w_ucb ranges from 0.2 (late) to 0.8 (early)
    w_ucb = 0.2 + 0.6 * exploration_factor
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
