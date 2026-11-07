# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # mu, sigma: arrays of shape (N,)
    # f_best: scalar, best value observed so far (lower is better)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Compute z-score for EI
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Check if we're running out of time (though this should be very fast)
    if time.time() > deadline:
        # Return the EI scores as-is (they're already computed)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
