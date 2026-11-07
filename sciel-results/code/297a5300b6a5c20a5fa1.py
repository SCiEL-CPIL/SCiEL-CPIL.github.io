import numpy as np
from scipy.special import erf

# Constants for EI calculation
SQRT_2PI = np.sqrt(2 * np.pi)
INV_SQRT_2 = 1 / np.sqrt(2)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Expected Improvement (EI).
    
    This function computes the Expected Improvement of evaluating a new point 
    compared to the current best value. For minimization, EI is defined as:
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given f(x) ~ N(mu(x), sigma(x)^2), the closed-form solution is:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    Phi is the standard normal CDF and phi is the standard normal PDF.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # A small epsilon is used for numerical stability
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Compute the standardized distance z
    z = (f_best - mu) / sigma_safe
    
    # Compute EI manually using erf for precision and speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    
    # Clip z to avoid numerical issues in exp for very large negative z (underflow to 0 is fine)
    # and very large positive z (erf goes to 1).
    # erf is stable for large inputs.
    z_erf_arg = z * INV_SQRT_2
    phi_z = 0.5 * (1.0 + erf(z_erf_arg))
    
    # Calculate exp term. For large |z|, exp(-0.5*z^2) underflows to 0, which is correct.
    # We can clip z for the exp calculation to prevent warnings, though numpy handles it.
    z_sq_half = 0.5 * z * z
    # exp(-1000) is 0.
    pdf_z = np.exp(-z_sq_half) / SQRT_2PI
    
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability might cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
