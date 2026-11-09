"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _compute_z(mu, sigma, f_best):
    """Compute standardized improvement z."""
    sigma_safe = np.maximum(sigma, 1e-9)
    return (f_best - mu) / sigma_safe


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized acquisition function blending EI and UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition values (higher = more promising)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = _compute_z(mu, sigma, f_best)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # UCB component: optimistic lower bound
    # Use a fixed kappa for exploration
    kappa = 1.5
    ucb = mu - kappa * sigma
    
    # Blend EI and UCB: weight EI more heavily for exploitation, UCB for exploration
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_range > 0 else 1.0
    
    # Normalize UCB to [0, 1] range (lower mu, higher sigma -> higher score)
    ucb_norm = (ucb_max - ucb) / ucb_range  # inverts so lower ucb -> higher score
    ucb_norm = np.clip(ucb_norm, 0, 1)
    
    # EI normalized to [0, 1]
    ei_norm = ei / ei_max if ei_max > 0 else np.zeros_like(ei)
    
    # Blend: 60% EI, 40% UCB
    alpha = 0.6
    blended = alpha * ei_norm + (1 - alpha) * ucb_norm
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    scores = blended / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
