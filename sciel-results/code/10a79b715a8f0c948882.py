"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    For each candidate x with predictive distribution N(mu(x), sigma(x)^2),
    EI(x) = E[max(0, f_best - f(x))]
          = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    This directly optimizes the expected reduction in the current best value,
    which aligns with the scoring metric (simple regret).
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores for each candidate. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # EI is non-negative by construction, but clip just in case
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
