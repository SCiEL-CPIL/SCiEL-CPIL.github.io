import numpy as np
import time
from scipy.special import ndtr

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure sigma is strictly positive and avoid extreme values
    sigma = np.clip(sigma, 1e-10, 1e6)
    
    # Calculate the z-score for improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow in exp
    z_clipped = np.clip(z, -50, 50)
    
    # Standard normal CDF and PDF
    Phi_z = ndtr(z_clipped)
    phi_z = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
