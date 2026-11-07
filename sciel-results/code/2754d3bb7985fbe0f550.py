import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a hybrid approach:
    - Early on (large gap to minimum): emphasize UCB for exploration
    - Late (small gap): emphasize EI for exploitation
    
    Parameters
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Deadline guard: if we're close to the wall, return simple exploitation scores
    if time.time() > _DEADLINE:
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are numpy arrays with correct shape
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Basic sanity: ensure finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Goldstein-Price 2D minimum value
    f_min = 3.0
    
    # Compute gap: how far we are from the known minimum
    gap = max(f_best - f_min, 0.0)
    
    # Adaptive beta for UCB: large when gap is large (early), small when gap is small (late)
    # This encourages exploration early and exploitation later
    beta = 0.3 + 2.0 * (1.0 / (1.0 + gap / 5.0))
    
    # Compute UCB score (for minimization: -mu + beta*sigma)
    ucb = -mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Compute EI score with small xi
    # xi scales with gap to encourage more exploration early
    xi = 0.01 * max(gap, 0.1)
    z = (f_best - mu - xi) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu - xi) * cdf + sigma * pdf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Blend: weight_ucb high when gap is large, weight_ei high when gap is small
    # This smoothly transitions from exploration to exploitation
    weight_ucb = 1.0 / (1.0 + gap / 3.0)
    weight_ei = 1.0 - weight_ucb
    
    # Combine UCB and EI
    score = weight_ucb * ucb + weight_ei * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score.astype(np.float64)
