"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_scores(mu, sigma, f_best, elapsed_start):
    """
    Computes acquisition scores using a pure Gaussian Process Lower Confidence Bound (GP-LCB).
    
    For minimization, the Lower Confidence Bound is L(x) = mu(x) - kappa * sigma(x).
    We want to find x that minimizes L(x), which is equivalent to maximizing -L(x).
    Score = -mu(x) + kappa * sigma(x).
    
    This structural change removes the Expected Improvement component and the 
    normalization/blending, relying solely on the LCB to balance exploration 
    (high sigma) and exploitation (low mu).
    """
    # Ensure sigma is non-negative and safe for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Exploration parameter. 
    # A higher kappa encourages more exploration. 
    # The directive suggests 2.0 as an example. 
    # Given the multimodal nature of Styblinski-Tang and the limited budget (30 evals),
    # aggressive exploration is often necessary to locate the global basin.
    kappa = 2.0
    
    # Pure GP-LCB score for minimization
    # We maximize this score.
    score = -mu + kappa * sigma_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
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
    
    # Compute pure LCB scores
    scores = _compute_scores(mu, sigma, f_best, t0)
    
    # Deadline guard: if we've used too much time, return what we have
    # The computation is very fast, so this is mostly a safety net.
    elapsed = time.time() - t0
    if elapsed > 325.0:
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
