"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    # Deadline Guard
    if time.time() - _bo_start_time > 305:
        return -np.asarray(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Expected Improvement (EI) for minimization
    # Improvement = f_best - f(x)
    # We want to maximize E[Improvement]
    
    # Handle sigma == 0 to avoid division by zero
    # If sigma is 0, EI is 0 if mu < f_best, else 0 (or just 0)
    # Actually if sigma=0, the distribution is a point mass.
    # EI = (f_best - mu) * 1 if mu < f_best else 0
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    # Phi(z) is CDF, phi(z) is PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # To encourage exploration in early stages or if EI is very small,
    # we might add a small UCB component?
    # No, let's keep it pure EI first as a structural change from TS.
    # However, pure EI can be very greedy.
    # Let's add a small exploration term to escape local minima.
    # Hybrid: EI + alpha * sigma
    # Or just EI.
    
    # Let's try a hybrid: 0.9 * EI + 0.1 * (-mu + 2*sigma) ?
    # No, let's stick to EI but with a small xi (exploitation parameter)
    # EI_xi = (f_best - xi - mu) * Phi(z_xi) + sigma * phi(z_xi)
    # z_xi = (f_best - xi - mu) / sigma
    
    xi = 0.01
    z_xi = (f_best - xi - mu) / sigma_safe
    cdf_xi = norm.cdf(z_xi)
    pdf_xi = norm.pdf(z_xi)
    
    ei_xi = (f_best - xi - mu) * cdf_xi + sigma_safe * pdf_xi
    
    # Ensure finite
    ei_xi[~np.isfinite(ei_xi)] = 0.0
    
    return ei_xi
# EVOLVE-BLOCK-END
