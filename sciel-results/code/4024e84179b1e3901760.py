"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time to enforce deadline
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > 300:
        # Fallback: return something reasonable, e.g., UCB with beta=0 (pure exploitation)
        # Or just return -mu
        return -mu

    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma^2(x))
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Compute EI
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # For points where sigma is very small, EI might be negative if mu > f_best
    # We should only select points with positive EI, but the problem says "higher meaning evaluate next"
    # So we can return EI directly. Negative EI means no improvement expected.
    
    # Ensure finite output
    scores = np.where(np.isfinite(ei), ei, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
