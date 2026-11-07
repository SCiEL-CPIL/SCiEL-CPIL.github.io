# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf, erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with exploration bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    sqrt2 = np.sqrt(2.0)
    
    # Phi(z) = 0.5 * erfc(-z / sqrt2)
    Phi = 0.5 * erfc(-z / sqrt2)
    
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    phi = np.exp(-0.5 * z * z - 0.5 * np.log(2.0 * np.pi))
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add exploration bonus: prefer points with higher uncertainty
    # This helps escape local optima in the acquisition function landscape
    exploration_bonus = 0.1 * sigma
    
    # Combine EI with exploration bonus
    score = ei + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
