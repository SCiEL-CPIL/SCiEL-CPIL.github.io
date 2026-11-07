# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses pure Expected Improvement (EI).
    
    EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x)) / sigma(x)
    
    This focuses strictly on the expected magnitude of improvement, which is
    optimal for smooth, unimodal functions like the sphere once the initial
    exploration phase is complete.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize inputs
    # Ensure sigma is strictly positive to avoid division by zero
    # A very small epsilon prevents numerical issues without significantly altering the posterior
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)

    # Calculate Z-score
    # Z = (f_best - mu) / sigma
    # Clip Z to prevent extreme values in erf/exp which can lead to loss of precision or overflow
    # erf is well-behaved, but for large |Z|, Phi(Z) approaches 0 or 1.
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)

    # Compute Phi(Z) and phi(Z) using standard normal CDF and PDF
    # Phi(Z) = 0.5 * (1 + erf(Z / sqrt(2)))
    # phi(Z) = (1 / sqrt(2*pi)) * exp(-Z^2 / 2)
    
    sqrt_2 = np.sqrt(2.0)
    # Use np.erf for stability
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi_Z = 0.5 * (1.0 + np.erf(z / sqrt_2))
    
    # Compute EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: (f_best - mu) can be negative if mu > f_best, but then Phi(Z) is small.
    # However, mathematically EI is always >= 0.
    # To ensure non-negativity against floating point errors:
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return np.zeros_like(score)
        
    return score
# EVOLVE-BLOCK-END
