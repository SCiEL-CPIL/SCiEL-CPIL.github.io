import numpy as np
import time
from scipy.special import ndtr, ndtri

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    """
    # Track elapsed time to ensure we finish within the deadline
    start_time = time.time()
    deadline = 300.0  # Leave margin from 330s wall-clock limit
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate the z-score for improvement
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF using scipy.special for speed
    Phi_z = ndtr(z)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        # If we're too close to the deadline, just return what we have
        pass
    
    return ei
