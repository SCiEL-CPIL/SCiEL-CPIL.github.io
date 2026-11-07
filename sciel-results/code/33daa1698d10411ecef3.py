import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses a combination of Expected Improvement (EI) and UCB-style exploration,
    with adaptive beta based on the spread of sigma relative to the improvement potential.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : ndarray, shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        Best observed function value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores; higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT - _SAFETY_MARGIN:
        # Fallback: pure exploitation (pick lowest mean)
        return f_best - mu

    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(sigma, 1e-10, None)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-style exploration term
    # Use an adaptive beta that balances exploration and exploitation
    # beta based on the quantile of sigma: use median sigma as scale
    sigma_median = np.median(sigma)
    if sigma_median < 1e-8:
        sigma_median = 1e-8
    
    # Adaptive beta: larger when uncertainty is high relative to improvement potential
    # Use a fixed beta but scale by the relative uncertainty
    beta = 1.5
    
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weighted combination: favor EI more as we have more data, but keep exploration
    # Use a fixed weight of 0.7 for EI and 0.3 for UCB
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
