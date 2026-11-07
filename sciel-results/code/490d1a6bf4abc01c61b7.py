"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for Bayesian optimization using Probability of Improvement (PI).
    
    PI is a structurally different acquisition function from Expected Improvement (EI).
    While EI maximizes the expected magnitude of improvement, PI maximizes the probability
    that the candidate point improves upon the current best. This can lead to more robust
    exploration in the early stages of optimization, as it is less sensitive to the absolute
    scale of the function values and focuses on the likelihood of finding a better point.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (Probability of Improvement) for each candidate point.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.clip(sigma, 1e-9, None)
    
    # Compute z-score for Probability of Improvement
    # P(f(x) < f_best) = P((f(x) - mu) / sigma < (f_best - mu) / sigma)
    #                   = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_clipped
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(pi), pi, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
