"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best observed (greedy) or zeros
        # Greedy is safe
        return -mu

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This is the expected improvement over the current best
    
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Compute EI using standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite
    if not np.all(np.isfinite(ei)):
        ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return ei
# EVOLVE-BLOCK-END
