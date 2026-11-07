"""Rewritten acquisition for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timing guard
_ACQ_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Variance-Gated Expected Improvement (VEI).
    
    Combines EI with a variance-gating mechanism that penalizes exploitation
    in low-uncertainty regions, forcing the search to prioritize areas where
    the GP is still uncertain. This helps escape local minima in Branin's
    multimodal landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed objective value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    # Check time budget
    elapsed = time.time() - _ACQ_START
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: pick the point with lowest mu
        mu_arr = np.asarray(mu)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance-Gating: scale EI by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        # gamma=1.5 provides a strong but not extreme penalty
        gamma = 1.5
        variance_gate = (sigma_safe / sigma_max) ** gamma
    else:
        variance_gate = np.ones_like(sigma_safe)
    
    # Variance-Gated EI
    vei = ei * variance_gate
    
    # Add a small UCB component for additional exploration
    # UCB for minimization: -mu + beta * sigma
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize UCB to [0, 1] for a smooth exploration bonus
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > eps:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine: VEI as primary, with a small UCB bonus
    # The UCB bonus helps differentiate among candidates with similar VEI
    score = vei + 0.1 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
