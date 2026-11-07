"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Score(x) = -mu(x) + beta * sigma(x)
    
    We use a dynamic beta that increases with f_best to encourage more
    exploration when the current best is poor, and less when it is good.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        UCB scores. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    # Dynamic beta: 
    # Base exploration level 1.0, up to 2.0 additional when f_best is large.
    # f_best for Ackley ranges from ~0 to ~20.
    # beta = 1.0 + 1.0 * (f_best / (1.0 + f_best))
    # At f_best=0, beta=1.0. At f_best=1, beta=1.5. At f_best=5, beta~1.83.
    beta = 1.0 + 1.0 * (f_best / (1.0 + f_best))
    
    # UCB for minimization: we want to minimize mu - beta*sigma.
    # So we maximize -mu + beta*sigma.
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
