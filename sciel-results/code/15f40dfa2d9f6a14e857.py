# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a blend of Expected Improvement (EI) and a tuned UCB-like term
    to balance exploitation and exploration. A time guard ensures we
    return valid scores before the wall-clock limit.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Std of the GP posterior.
    f_best : float
        Best observed value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher = more promising.
    """
    # Deadline guard: return safe finite scores if near the limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Z-score for EI (minimization)
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB term: -mu + beta * sigma
    # Tune beta to encourage exploration early, exploitation later
    # Use a moderate beta that adapts to the scale of sigma
    beta = 0.6
    ucb = -mu + beta * sigma
    
    # Normalize both components to zero-mean, unit-variance
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-12
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Blend: weight EI more for exploitation, UCB for exploration
    w_ei = 0.65
    w_ucb = 0.35
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
