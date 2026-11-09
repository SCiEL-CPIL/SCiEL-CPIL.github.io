# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants
_N_CAND = 2000

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid of probability-of-significant-improvement (PSI)
    and expected improvement per unit cost, with exploration bonus.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Guard against zero or negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Significant Improvement (PSI)
    # P(f(x) < f_best - delta) for some small delta
    # This naturally decays as f_best approaches the minimum
    delta = max(0.01 * abs(f_best), 1e-6)
    z_psi = (f_best - delta - mu) / sigma_safe
    psi = norm.cdf(z_psi)
    
    # Cost-aware expected improvement
    ei_per_cost = ei / cost_safe
    
    # Combine: weighted blend of PSI (exploration) and EI per cost (exploitation)
    # As f_best gets close to 0, the scale of improvement shrinks,
    # so we weight PSI more to maintain exploration pressure
    f_best_abs = abs(f_best)
    # Weight: when f_best is large, favor EI/cost; when small, favor PSI
    w_psi = 0.3 + 0.7 * (1.0 / (1.0 + f_best_abs))
    w_ei = 1.0 - w_psi
    
    # Normalize components to comparable scale
    # PSI is in [0, 1], EI/cost can vary widely
    max_ei_per_cost = np.max(ei_per_cost)
    if max_ei_per_cost > 1e-12:
        ei_per_cost_norm = ei_per_cost / max_ei_per_cost
    else:
        ei_per_cost_norm = ei_per_cost
    
    scores = w_psi * psi + w_ei * ei_per_cost_norm
    
    # Add a small exploration bonus for high uncertainty regions
    # normalized by typical sigma to avoid scale issues
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        exploration_bonus = 0.05 * (sigma_safe / max_sigma)
    else:
        exploration_bonus = 0.0
    
    scores = scores + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
