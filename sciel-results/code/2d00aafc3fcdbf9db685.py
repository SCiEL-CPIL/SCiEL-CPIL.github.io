"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track number of evaluations for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: if we're running long, just return something safe
    # (though with 2000 points and simple math, this shouldn't be an issue)
    
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is CDF and phi is PDF of standard normal
    # Higher EI means more improvement expected
    
    # Handle sigma = 0 case
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Z-score for EI calculation
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: for minimization, we want to find points where mu < f_best
    # The standard EI formula for minimization is:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure non-negative EI (numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
