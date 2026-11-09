"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri
import time

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D Weierstrass.
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard: if we are too close to the time limit, return a safe, fast result
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback to a simple exploitation score
        return -mu
    
    # Ensure inputs are 1D arrays of floats
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    # Use scipy.special for numerical stability
    # ndtr is the standard normal CDF (Phi)
    # pdf = exp(-z^2/2) / sqrt(2*pi)
    cdf_val = ndtr(z_clipped)
    pdf_val = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement_term = (f_best - mu) * cdf_val
    exploration_term = sigma_safe * pdf_val
    ei_scores = improvement_term + exploration_term
    
    # Add a small UCB-like exploration bonus to encourage more exploration
    # in early stages, scaled by uncertainty
    # Use a moderate kappa that adapts slightly with the number of evaluations
    # Since we don't know the step count directly, use a fixed but smaller kappa
    # to complement EI's built-in exploration
    kappa = 0.5
    ucb_bonus = kappa * sigma_safe
    
    # Combine EI and UCB bonus
    scores = ei_scores + ucb_bonus
    
    # Ensure finite output - replace any non-finite values with a very negative score
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
