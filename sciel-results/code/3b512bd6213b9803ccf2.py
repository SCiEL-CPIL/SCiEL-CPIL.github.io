"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses a Probability of Improvement (POI) based acquisition function 
scaled by the expected gain magnitude. This structural change replaces 
the standard EI's PDF-based exploration term with a POI-scaled 
uncertainty component, aiming to improve exploration-exploitation 
balance in the 30-evaluation budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
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
        # Fallback: Simple UCB with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z score: (f_best - mu) / sigma
    # If mu < f_best, z is positive (improvement likely)
    # If mu > f_best, z is negative (improvement unlikely)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical underflow/overflow in CDF
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Probability of Improvement (POI)
    # POI = Phi(Z)
    poi = norm.cdf(z_clipped)
    
    # Ensure finite values
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    # Structural Change:
    # Standard EI: (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # This POI-based score: POI * (f_best - mu + alpha * sigma)
    # We use alpha = 0.5 to balance exploration.
    # This scales the "magnitude of expected gain" (f_best - mu) 
    # and the "uncertainty magnitude" (sigma) by the probability 
    # that the point is actually an improvement.
    
    alpha = 0.5
    scores = poi * (f_best - mu + alpha * sigma_safe)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Ensure non-negative? 
    # If f_best - mu is very negative and poi is small, score could be negative.
    # Since we want "higher meaning evaluate next", negative scores are fine 
    # as long as they are lower than positive ones. 
    # However, if all scores are negative, we pick the least negative.
    # EI is always non-negative. 
    # Let's check if we should clamp to 0.
    # If mu > f_best + alpha*sigma, then (f_best - mu + alpha*sigma) < 0.
    # POI is small. Score is small negative.
    # Points with mu < f_best will have positive scores.
    # This correctly prioritizes points predicted to be better.
    
    return scores
# EVOLVE-BLOCK-END
