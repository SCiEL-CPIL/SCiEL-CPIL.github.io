"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a hybrid of Probability of Improvement (PI) and Upper Confidence Bound (UCB)
to balance aggressive exploitation with necessary exploration for multi-modal landscapes.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid PI + UCB acquisition function.
    
    Structural Change: Replaces Expected Improvement (EI) with Probability of Improvement (PI)
    as the exploitation term, combined with a strong UCB term for exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to minimize f, so we prefer low mu and high sigma.
    # Score = -mu + beta * sigma
    # Beta is set to 2.0 to ensure sufficient exploration in a multi-modal function,
    # compensating for the exploitative nature of PI.
    beta = 2.0
    ucb_scores = -mu + beta * sigma_safe
    
    # 3. Normalize scores to combine them effectively
    # Standardize (z-score) to handle different scales of PI (0-1) and UCB (unbounded)
    def standardize(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < eps:
            return np.zeros_like(arr)
        return (arr - mean) / std
    
    pi_std = standardize(pi_scores)
    ucb_std = standardize(ucb_scores)
    
    # 4. Combine
    # Weighting: PI provides the direct exploitation signal (probability of being better).
    # UCB provides the exploration signal (uncertainty).
    # A 50/50 split is a robust starting point for multi-modal functions when using PI.
    w_pi = 0.5
    w_ucb = 0.5
    
    scores = w_pi * pi_std + w_ucb * ucb_std
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
