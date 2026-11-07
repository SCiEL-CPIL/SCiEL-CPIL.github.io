"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Knowledge Gradient (KG) acquisition for minimization
    # KG estimates the expected reduction in the global best value.
    # For a single observation at x: E[max(0, f_best - Y)] where Y ~ N(mu(x), sigma(x)^2)
    # This is exactly Expected Improvement (EI) for minimization.
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    # Calculate EI for minimization
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    d = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF using scipy for numerical stability
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    ei = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Clip negative EI to 0 (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
