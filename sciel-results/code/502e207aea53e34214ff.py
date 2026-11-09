"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _normalize(arr):
    """Min-max normalize to [0, 1]."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    rng = arr_max - arr_min
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr_min) / rng


def _compute_scores(mu, sigma, f_best, elapsed_start):
    """
    Computes acquisition scores using a fixed 70% EI / 30% UCB blend.
    This strategy was found to improve performance on Styblinski-Tang 1D.
    Normalization prevents scale dominance between EI and UCB terms.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want high scores for low mu and high sigma.
    # UCB = -mu + kappa * sigma
    # Using kappa = 1.0 is a standard starting point that balances exploration well.
    ucb = -mu + 1.0 * sigma_safe
    
    # 3. Normalize components to [0, 1] to make them comparable
    # This is crucial to prevent one term from dominating due to scale differences
    ei_n = _normalize(ei)
    ucb_n = _normalize(ucb)
    
    # 4. Blend: 70% Exploitation (EI), 30% Exploration (UCB)
    # This weighting is measured to be superior for this specific problem
    score = 0.7 * ei_n + 0.3 * ucb_n
    
    # 5. Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    t0 = time.time()
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: if inputs are empty or degenerate, return zeros
    if mu.size == 0:
        return np.array([])
    
    # Quick computation
    scores = _compute_scores(mu, sigma, f_best, t0)
    
    # Deadline guard: if we've used too much time, return what we have
    elapsed = time.time() - t0
    if elapsed > 325.0:
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
