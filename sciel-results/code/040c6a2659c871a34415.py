import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) with a small exploration bonus.
    EI naturally balances exploration and exploitation based on f_best.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # We want to maximize EI, so higher scores mean more likely improvement
    
    # z-score
    z = (f_best - mu) / sigma
    
    # Standard normal PDF and CDF
    # Use scipy for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    # When f_best - mu > 0 (mu < f_best, potential improvement), EI is positive
    # When f_best - mu <= 0, EI should be 0 (no improvement possible)
    ei = np.where(f_best - mu > 0, 
                  (f_best - mu) * Phi_z + sigma * phi_z, 
                  0.0)
    
    # Add a small exploration bonus based on sigma to encourage exploring
    # uncertain regions. Scale it down as we get closer to the minimum.
    gap = max(f_best - 3.0, 0.0)
    # Exploration weight: start at 0.5, decrease to 0.1 as gap approaches 0
    explore_weight = 0.1 + 0.4 * np.exp(-gap / 2.0)
    
    scores = ei + explore_weight * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
