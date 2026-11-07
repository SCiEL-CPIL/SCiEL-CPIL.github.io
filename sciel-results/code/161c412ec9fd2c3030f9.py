"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PoI) acquisition function with an exploration bias.
    
    This variant introduces a small negative offset (xi) to the improvement threshold,
    effectively broadening the region of candidates considered for exploration.
    This helps in escaping local optima in multi-modal landscapes like Branin by
    giving higher scores to uncertain regions that have a non-negligible probability
    of containing a global minimum, even if the GP mean is slightly worse than f_best.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration parameter: negative xi shifts the threshold to be less strict,
    # encouraging exploration of regions with high uncertainty.
    xi = -0.5
    
    # Adjusted z-score: P(f(x) < f_best + xi)
    # Since xi is negative, f_best + xi < f_best, so we are asking for
    # probability of being better than a value *worse* than current best?
    # Wait.
    # Standard PoI: P(f(x) < f_best) -> z = (f_best - mu) / sigma
    # If we use threshold T = f_best + xi:
    # P(f(x) < T) -> z = (T - mu) / sigma = (f_best + xi - mu) / sigma
    # If xi = -0.5, T = f_best - 0.5.
    # We are asking for probability that f(x) < f_best - 0.5.
    # This is *harder* to achieve than f(x) < f_best.
    # So this is MORE exploitative.
    
    # To be MORE exploratory, we want T > f_best.
    # So we should use xi > 0 in the formula z = (f_best + xi - mu) / sigma.
    # Let's use xi = 0.5.
    # Then we ask for P(f(x) < f_best + 0.5).
    # This is easier to satisfy, so scores are higher for more points.
    # This broadens the search.
    
    xi = 0.5
    z = (f_best + xi - mu) / sigma_safe
    
    # Probability of Improvement: Phi(z)
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
