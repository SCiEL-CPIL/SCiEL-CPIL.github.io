"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a hybrid approach: primarily Expected Improvement (EI) but with
    a safety margin to avoid numerical issues, and a small exploration bonus
    based on uncertainty to escape local optima.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # For minimization, we want high scores where mu is low (good) and sigma is high (uncertain)
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical edge cases
    # When z is very large positive, cdf(z) ~ 1, pdf(z) ~ 0, so ei ~ (f_best - mu)
    # When z is very large negative, cdf(z) ~ 0, pdf(z) ~ 0, so ei ~ 0
    
    # Add a small uncertainty-based exploration term to encourage exploration
    # This helps escape local optima in the Styblinski-Tang function
    # Scale the exploration term to be small relative to EI
    exploration = 0.1 * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
