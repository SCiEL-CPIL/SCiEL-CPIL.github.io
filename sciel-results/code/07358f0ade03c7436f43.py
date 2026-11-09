"""Rewritten program using a normalized EI + UCB blend."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _minmax_normalize(x):
    """Normalize an array to [0, 1] range. Handles edge cases."""
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-9:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    normalized Upper Confidence Bound (UCB) term.
    
    This approach normalizes both components to [0, 1] before blending,
    ensuring that the relative contribution of exploitation (EI) and
    exploration (UCB) is controlled by the weights rather than the raw
    scales of the posterior statistics.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: Lower Confidence Bound (LCB) style for minimization
    # We want to maximize the score, so we use mu - beta * sigma.
    # Low mu is good, high sigma is good for exploration.
    beta = 2.0
    ucb = mu - beta * sigma_safe
    
    # Normalize both components to [0, 1]
    ei_norm = _minmax_normalize(ei)
    ucb_norm = _minmax_normalize(ucb)
    
    # Blend with 70% exploitation and 30% exploration
    w_exploit = 0.7
    w_explore = 0.3
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
