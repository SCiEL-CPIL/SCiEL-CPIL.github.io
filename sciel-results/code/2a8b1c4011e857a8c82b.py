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
    Computes acquisition scores using pure GP-Lower Confidence Bound (GP-LCB).
    score = -mu - kappa * sigma
    
    This directly maximizes the lower bound of the objective, encouraging
    exploration where uncertainty is high while exploiting regions with low
    predicted values.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # GP-LCB for minimization: we want to minimize mu + kappa * sigma
    # So the score (higher is better) is -mu - kappa * sigma
    kappa = 2.0
    
    score = -mu - kappa * sigma_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=-np.inf, posinf=1e10, neginf=-1e10)
    
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
