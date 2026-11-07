"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition function.
    
    This function combines the exploitation capability of EI with the exploration
    capability of UCB to handle the multimodal nature of the Weierstrass function.
    
    Parameters:
    mu (np.array): Posterior mean at candidate points, shape (N,)
    sigma (np.array): Posterior standard deviation at candidate points, shape (N,)
    f_best (float): Current best observed value
    
    Returns:
    np.array: Scores for each candidate, shape (N,). Higher is better.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: simple UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical issues in norm.pdf/cdf for extreme values
    z = np.clip(z, -10.0, 10.0)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) score for minimization
    # We want to maximize the score.
    # Low mu is good. High sigma is good for exploration.
    # UCB_score = -mu + beta * sigma
    beta = 2.0  # Exploration weight, tuned for multimodal landscape
    ucb = -mu + beta * sigma
    
    # 3. Combine EI and UCB
    # EI is non-negative and represents expected gain.
    # UCB represents the potential lower bound.
    # We sum them with a weight on UCB to balance exploration/exploitation.
    # A weight of 1.0 on EI and 0.5 on UCB is a reasonable starting point,
    # but given the success of "EI + UCB" in previous runs, we keep it simple.
    
    # Normalize EI slightly if it dominates? 
    # EI is in objective units. UCB is in objective units.
    # They are comparable.
    
    score = ei + 0.5 * ucb
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
