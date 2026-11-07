"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Check time budget - if running out, fall back to simple exploitation
    if time.time() > _DEADLINE - 5:
        return -mu
    
    # Structural change: use Expected Improvement with adaptive exploration
    # The key insight from the best inspiration (score 1.0) is that pure EI
    # with the exact formula works well. We'll use EI but add a small
    # exploration bonus scaled by uncertainty to handle early-stage better.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    
    # Core EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: add a small bonus proportional to sigma
    # This helps in early stages when EI might be too exploitative
    # The coefficient is small (0.1) to not disrupt the core EI signal
    # but provides a gentle push toward uncertain regions
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure numerical stability
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
