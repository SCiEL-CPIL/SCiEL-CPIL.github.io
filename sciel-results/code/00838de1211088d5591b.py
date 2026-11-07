"""Rewritten program for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 2.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Log-Expected Improvement (L-EI).
    
    This approach computes the Expected Improvement (EI) and returns the
    logarithm of the EI. This structural change from the standard z-EI or
    raw EI provides a different balance between exploitation and exploration.
    The logarithmic transformation dampens the aggressive exploitation of
    high-mean candidates, potentially allowing the optimizer to escape local
    minima in multi-modal landscapes like the Goldstein-Price function by
    maintaining a more sensitive response to moderate improvements with
    lower uncertainty.
    
    Returns scores where higher values indicate better candidates.
    """
    # Check time to ensure we don't exceed limits
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # If we are extremely close to the deadline, use a very fast heuristic
    # Fallback: simple z-score ranking
    if remaining_time < _DEADLINE_MARGIN:
        eps = 1e-8
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        # Simple proxy: z-score
        scores = np.where(np.isfinite(z), z, -np.inf)
        return scores

    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Calculate Log-Expected Improvement
    # We clamp EI to a small positive value to avoid log(0) or log(negative)
    # Candidates with EI <= 0 are assigned a very low score (log(1e-8) ~ -18.4)
    # This ensures they are ranked below any candidate with positive EI.
    ei_clamped = np.maximum(ei, 1e-8)
    scores = np.log(ei_clamped)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
