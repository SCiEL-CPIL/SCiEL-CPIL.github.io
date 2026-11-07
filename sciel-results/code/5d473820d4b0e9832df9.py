"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI balances exploration and exploitation by calculating the expected
    amount by which a candidate point might improve upon the current best
    value (f_best). This is generally more robust than fixed-kappa LCB
    in low-budget regimes because it naturally adapts the exploration
    weight based on the posterior uncertainty and the current best value.
    """
    # ----- time guard (safety margin) -----
    t0 = time.time()
    deadline = t0 + 290.0  # Leave 10s for safety

    n = mu.shape[0]
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # We are minimizing, so we look for improvement if mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    # Vectorized calculation
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    improvement = f_best - mu
    # If mu > f_best, improvement is negative, but EI should be small positive.
    # The formula handles this naturally.
    
    scores = improvement * cdf_z + sigma_safe * phi_z
    
    # EI is non-negative. Ensure finiteness.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we are dangerously close to the deadline, return immediately
    if time.time() > deadline - 0.5:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
