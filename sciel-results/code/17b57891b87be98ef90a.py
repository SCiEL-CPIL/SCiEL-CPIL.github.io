"""
Rewritten program using a deterministic LCB acquisition with a constant beta.
This approach provides a stable balance between exploration and exploitation,
avoiding the potential instability of adaptive coefficients.
"""
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
    # Leave a safety margin of 1.5s
    return max(0.0, 328.5 - _elapsed())

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) with a constant beta.
    For minimization, we maximize the score: -mu + beta * sigma.
    A constant beta ensures a consistent exploration/exploitation trade-off.
    """
    if _elapsed() > 328.5:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Constant beta for LCB
    # A value of 5.0 is chosen to provide robust exploration in the early stages
    # and sufficient exploitation in later stages for the Michalewicz 2D function.
    beta = 5.0
    
    # Calculate LCB scores: higher score means more promising
    # We want to minimize f, so we look for low mu.
    # We want to explore high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
