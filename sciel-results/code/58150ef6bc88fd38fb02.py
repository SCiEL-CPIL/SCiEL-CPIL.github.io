"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track number of evaluations
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard
    start_time = time.time()
    deadline = 330.0
    
    # Structural change: Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For minimization, higher EI means better candidate
    scores = ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() - start_time > deadline - 5.0:
        # Return safe fallback
        return np.full_like(mu, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
