"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) for minimization.
    Higher score = more likely to be evaluated.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe fallback: just exploit the best mean
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF of standard normal, phi is PDF of standard normal
    
    # Handle sigma near zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score: how far mu is below f_best in units of sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10.0, 10.0)
    
    # Phi(z) = standard normal CDF
    # phi(z) = standard normal PDF
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected improvement in objective units
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware: divide by cost
    scores = ei / cost
    
    # Safety check for finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
