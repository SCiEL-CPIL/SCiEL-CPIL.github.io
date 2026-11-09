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
    Computes acquisition scores using GP-LCB (Lower Confidence Bound).
    score = -mu - kappa * sigma, where higher score means more promising.
    This directly maximizes the lower bound of the objective.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # GP-LCB for minimization: we want to minimize the lower bound of f(x)
    # Lower bound estimate: mu(x) - kappa * sigma(x)
    # We want to MAXIMIZE this lower bound, so score = -(mu - kappa*sigma) = -mu + kappa*sigma
    # But the directive says score = -mu - kappa * sigma. Let me re-read...
    # "score = -mu - kappa * sigma" - this means high score when mu is very negative (good) 
    # and sigma is small (confident). Wait, that doesn't make sense for exploration.
    # 
    # Actually, for GP-LCB in minimization:
    # We minimize: L(x) = mu(x) - kappa * sigma(x)
    # So we want to find x that minimizes L(x), i.e., maximizes -L(x) = -mu(x) + kappa*sigma(x)
    # 
    # But the directive explicitly says: score = -mu - kappa * sigma
    # This would favor low mu AND low sigma (pure exploitation). Let me follow the directive exactly.
    kappa = 2.0
    score = -mu - kappa * sigma_safe
    
    # Ensure finite values
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
