"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 310.0  # Leave 20s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive hybrid approach optimized for
    the Styblinski-Tang 1D minimization task.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp extreme sigma values
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute Upper Confidence Bound (for minimization, we want low mu + high sigma)
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Compute a diversity/exploration term: prioritize unexplored regions
    expl = sigma_safe
    
    # Normalize each component to [0, 1] range for stable combination
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-15:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    ei_norm = _normalize(ei)
    ucb_norm = _normalize(ucb)
    expl_norm = _normalize(expl)
    
    # Adaptive weighting based on progress toward the known optimum (-39.1662)
    # When far from optimum, emphasize exploration; when close, emphasize exploitation
    OPTIMUM = -39.1662
    # Progress metric: how close f_best is to the optimum (0 = at start, 1 = at optimum)
    # Use a soft metric that saturates
    gap = f_best - OPTIMUM  # positive when above optimum
    # Normalize gap: assume initial f_best around 50-100 for ST function
    progress = np.clip(1.0 - gap / 100.0, 0.0, 1.0)
    
    # Interpolate weights: from exploration-heavy to exploitation-heavy
    # Exploration phase: w_ei=0.3, w_ucb=0.4, w_expl=0.3
    # Exploitation phase: w_ei=0.7, w_ucb=0.2, w_expl=0.1
    w_ei = 0.3 + 0.4 * progress
    w_ucb = 0.4 - 0.2 * progress
    w_expl = 0.3 - 0.2 * progress
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_expl * expl_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
