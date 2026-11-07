"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pure Expected Improvement (EI) acquisition function.
    This structural change removes the heuristic UCB mixing and normalization
    which degraded performance in the parent lineage. Pure EI is a robust
    baseline that focuses on exploiting the GP posterior's prediction of
    improvement over the current best.
    
    mu, sigma: arrays of shape (2000,) in objective units.
    f_best: lowest observed value.
    Returns 2000 finite scores, higher = evaluate next.
    """
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical underflow/overflow in PDF/CDF
    # For z < -10, Phi(z) ~ 0 and phi(z) ~ 0, so EI ~ 0
    # For z > 10, Phi(z) ~ 1 and phi(z) ~ 0, so EI ~ f_best - mu
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is theoretically non-negative, but numerical errors can cause small negatives
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
