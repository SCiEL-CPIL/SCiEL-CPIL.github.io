"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimization.
    
    For minimization, EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    We return EI scores; higher means more likely to improve, so the
    evaluator picks the candidate with highest EI.
    """
    global _start_time, _DEADLINE
    
    # Deadline guard: if we're getting close to timeout, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return LCB as fallback (fast, deterministic)
        beta = 2.0
        return -mu + beta * sigma
    
    # Compute EI for minimization
    # Handle cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Use scipy for normal CDF and PDF
    from scipy.stats import norm
    
    # Phi(z) = norm.cdf(z), phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
