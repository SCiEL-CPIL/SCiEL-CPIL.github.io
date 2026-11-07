import numpy as np
from scipy.special import erf
import time

# Constants for EI calculation
SQRT_2PI = np.sqrt(2 * np.pi)
INV_SQRT_2 = 1 / np.sqrt(2)

# Global time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Noisy Expected Improvement (EI).
    Includes a deadline guard to ensure the program finishes within the time limit.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: If we are very close to the 330s limit, return a simple, fast score
    # to avoid timeout. We use 325s as a safety margin.
    if time.time() - _start_time > 325.0:
        # Return sigma (exploration) or f_best - mu (exploitation)?
        # Exploration is safer to avoid getting stuck if we are in a bad local optima.
        # But sigma might be small.
        # Let's return a simple UCB-like score: -mu + sigma
        return -mu + sigma

    # Ensure sigma is strictly positive to avoid division by zero
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Noisy EI parameter
    # A small noise parameter encourages exploration
    xi = 0.005
    
    # Compute the standardized distance z for Noisy EI
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    z_erf_arg = z * INV_SQRT_2
    phi_z = 0.5 * (1.0 + erf(z_erf_arg))
    
    # Calculate exp term
    z_sq_half = 0.5 * z * z
    pdf_z = np.exp(-z_sq_half) / SQRT_2PI
    
    # Noisy EI
    ei = (f_best - xi - mu) * phi_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
