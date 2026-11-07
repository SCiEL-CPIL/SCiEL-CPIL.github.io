"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 300.0  # seconds, with safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Pareto-filtered Expected Improvement (EI).
    
    Strategy:
    1. Mask candidates where mu > f_best (predicted mean is worse than current best).
       This restricts the search to regions where there is a non-zero probability of improvement.
    2. Apply Expected Improvement (EI) on the remaining candidates.
    3. Set masked candidates to a very low score (e.g., -inf or 0) so they are not selected.
    
    This structural change leverages the observation that Pareto filtering reduces wasted
    evaluations on poor areas, allowing the acquisition function to focus on promising regions.
    """
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5.0:
        # Fallback: simple exploitation
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        mu = np.where(np.isfinite(mu), mu, f_best)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical issues
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Pareto filter: Only consider candidates where mu < f_best
    # This ensures we are only looking at points that have a chance of being better than f_best
    mask = mu < f_best
    
    # Initialize scores to a very low value for masked points
    scores = np.full_like(mu, -np.inf)
    
    # Compute EI for the masked (valid) points
    if np.any(mask):
        mu_valid = mu[mask]
        sigma_valid = sigma[mask]
        
        # Calculate Expected Improvement (EI) for minimization
        # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
        z = (f_best - mu_valid) / sigma_valid
        
        # Clip z to avoid numerical overflow in exp
        z = np.clip(z, -10, 10)
        
        # Phi(z) and phi(z)
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        improvement = f_best - mu_valid
        ei = improvement * Phi_z + sigma_valid * phi_z
        
        # Assign EI scores to the valid points
        scores[mask] = ei
    
    # Ensure finite output: replace -inf with 0 or a small negative number
    # Since higher is better, we want masked points to have the lowest possible score
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
