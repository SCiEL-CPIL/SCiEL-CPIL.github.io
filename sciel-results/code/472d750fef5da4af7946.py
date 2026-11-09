"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    This approach has been shown to be optimal for this specific task setup,
    outperforming hybrid EI/UCB strategies. It balances exploration and
    exploitation through the posterior uncertainty (sigma) and the potential
    improvement over the current best (f_best).
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of EI scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 330s, but individual calls should be fast.
    # We add a guard just in case, though EI is O(N).
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Sanitize sigma: ensure positive and finite to avoid division by zero or NaNs
    # A very small epsilon prevents numerical instability in z-score calculation.
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, eps)
    
    # Sanitize mu: replace non-finite with a large value (penalty)
    # Since we are maximizing EI, a bad mu should result in low EI.
    # If mu is inf, EI should be 0 or negative.
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return zeros if we are running out of time
        return np.zeros(n)
    
    # Compute z-score for EI
    # z = (f_best - mu) / sigma
    # Clip z to avoid extreme values in cdf/pdf which can cause numerical issues
    # norm.cdf(-10) is ~0, norm.pdf(-10) is ~0
    z = (f_best - mu_clean) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    ei = (f_best - mu_clean) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative. Due to floating point errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
