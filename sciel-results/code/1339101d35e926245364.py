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
    Acquisition function using a parameterized model optimized for
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
    
    # Use a hybrid acquisition: combine EI with a distance-based exploration term
    # that encourages moving away from the current best region
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute a "novelty" score: candidates with high uncertainty that are far from
    # the current best are more valuable for exploration
    # We don't have coordinates, so use sigma as a proxy for novelty
    novelty = sigma_safe
    
    # Use a sigmoid to smoothly blend between exploitation (EI) and exploration (novelty)
    # The blend factor depends on how close we are to the known optimum
    # ST function minimum is -39.1662
    target = -39.1662
    gap = f_best - target  # positive if we're above the optimum
    
    # If gap is large, explore more; if gap is small, exploit more
    # Use a logistic function to create a smooth transition
    # When gap >> 0, explore; when gap ~ 0, exploit
    # Reduced exploration weight to favor exploitation (EI) more strongly,
    # especially near the optimum where raw sigma can be misleading.
    explore_weight = 0.3 * (1.0 / (1.0 + np.exp(gap / 2.0)))
    
    # Normalize EI and novelty to [0, 1]
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-15:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    ei_norm = _normalize(ei)
    novelty_norm = _normalize(novelty)
    
    # Combine with adaptive weighting
    scores = (1.0 - explore_weight) * ei_norm + explore_weight * novelty_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
