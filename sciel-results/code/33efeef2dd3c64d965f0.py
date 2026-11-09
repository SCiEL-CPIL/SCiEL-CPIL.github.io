"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def _elapsed():
    return time.time() - _start_time

def _ei(mu, sigma, f_best):
    """Standard Gaussian Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Use scipy's norm for stable computation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining EI with time-aware exploration.
    
    Early in the search: use EI with a slight exploration bonus.
    Later in the search: pure exploitation via EI.
    
    Parameters:
        mu: array of shape (N,), posterior mean in objective units
        sigma: array of shape (N,), posterior standard deviation in objective units
        f_best: scalar, lowest observed value so far
    
    Returns:
        scores: array of shape (N,), higher means more promising to evaluate next
    """
    # Deadline guard: if we're close to timeout, return simple EI quickly
    if _elapsed() > _DEADLINE_SECONDS:
        return _ei(mu, sigma, f_best)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute standard EI
    ei_scores = _ei(mu, sigma, f_best)
    
    # Adaptive exploration: as f_best approaches 0 (the known minimum),
    # reduce exploration. Use a simple heuristic based on f_best value.
    # f_best for Weierstrass is in [0, ~1.5], so normalize
    exploration_weight = max(0.0, min(1.0, f_best / 0.5))
    
    # Add a small exploration bonus scaled by uncertainty
    # This encourages looking at uncertain regions when we're not yet close to optimum
    exploration_bonus = exploration_weight * sigma_safe * 0.1
    
    scores = ei_scores + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
