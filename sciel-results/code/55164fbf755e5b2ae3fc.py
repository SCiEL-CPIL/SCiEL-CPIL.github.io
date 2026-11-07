"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Kriging Believer with adaptive exploration coefficient.
Instead of adding an exploration bonus, we shift the surrogate mean by
k*sigma, making exploration an intrinsic property of the surrogate model.
This avoids scale-mismatch between exploitation and exploration terms.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Kriging Believer.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple Kriging Believer with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 2.0 * sigma_safe)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Define reference values for the Styblinski-Tang function on [-5, 5]
    f_min_ref = -39.166
    f_max_ref = 40.0
    
    # Calculate progress: 0.0 at start (f_best ~ 40), 1.0 at end (f_best ~ -39)
    progress = (f_max_ref - f_best) / (f_max_ref - f_min_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive exploration coefficient for Kriging Believer
    # Start with high k to aggressively explore uncertain regions and find the global basin.
    # Decay to low k to refine the solution near the optimum.
    # Using a smooth decay that emphasizes exploration early on.
    k_start = 6.0
    k_end = 0.3
    # Smooth decay: k = k_end + (k_start - k_end) * exp(-3 * progress)
    # This gives rapid initial decay then plateaus
    k = k_end + (k_start - k_end) * np.exp(-3.0 * progress)
    
    # Kriging Believer: shift the surrogate mean by k * sigma
    # For minimization, we want to be optimistic, so we subtract k*sigma from mu
    # The shifted mean represents "what would the function value be if uncertainty
    # resolved optimistically"
    mu_shifted = mu - k * sigma_safe
    
    # Score: higher is better for selection, so we negate the shifted mean
    # (since we're minimizing)
    scores = -mu_shifted
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
