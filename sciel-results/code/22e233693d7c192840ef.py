# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pure Expected Improvement acquisition for minimizing Branin 2D.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        EI scores; higher means more promising to evaluate next.
    """
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
