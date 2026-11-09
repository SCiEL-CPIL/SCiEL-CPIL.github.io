"""Rewritten program using adaptive LCB (UCB for minimization) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _remaining():
    # Leave a safety margin
    return max(0.0, 328.5 - _elapsed())

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive Lower Confidence Bound (LCB).
    For minimization, we maximize the score: -mu + beta * sigma.
    beta is adapted based on f_best to shift from exploration to exploitation.
    """
    if _elapsed() > 328.5:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive beta
    # As f_best decreases (improves), we want to reduce exploration (lower beta).
    # f_best starts around 1.0-2.0 and goes to -1.8.
    # Let's use an exponential decay based on f_best.
    # If f_best is high (bad), beta is high. If f_best is low (good), beta is low.
    # Clip f_best to avoid extreme exponentials.
    f_best_clipped = np.clip(f_best, -2.0, 3.0)
    beta = 4.0 * np.exp(f_best_clipped)
    
    # Base score: UCB for minimization
    # We want to maximize this score.
    # High sigma -> high score (Explore)
    # Low mu -> high score (Exploit)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
