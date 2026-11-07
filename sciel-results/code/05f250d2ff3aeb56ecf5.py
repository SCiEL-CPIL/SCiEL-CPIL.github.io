import numpy as np
from scipy.stats import norm
import time

# Cost-aware acquisition function for Bayesian optimization on Levy 2D.
# Structural change: use a hybrid acquisition that combines:
#   1. Expected Improvement per unit cost (exploitation)
#   2. A distance-to-best penalty to encourage moving away from already-evaluated regions
#   3. An adaptive exploration term based on the relative uncertainty
# We also add a simple local refinement heuristic: if the GP mean is very close to f_best
# and sigma is small, we penalize that region to push search toward unexplored areas.

_START_TIME = None
_DEADLINE_SAFETY = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: hybrid EI/cost + exploration + distance penalty.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition values (higher = better)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    # If we're close to the deadline, return a simple score quickly
    if elapsed > 320.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Compute z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost: divide by cost to get EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    base_score = ei / cost_safe
    
    # Exploration term: reward high uncertainty, but normalized
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    exploration = 0.1 * (sigma_safe / sigma_max)
    
    # Penalty for regions where the GP is very confident and close to f_best
    # This encourages the search to explore new regions
    # mu_close = how close mu is to f_best (lower is closer)
    mu_diff = np.abs(mu - f_best)
    # If mu is very close to f_best and sigma is small, this is a "known good" region
    # We want to penalize it to push exploration
    # Use a sigmoid-like penalty
    penalty_scale = 1.0  # scale factor for the penalty
    known_good = (mu_diff < 0.01 * max(abs(f_best), 1.0)) & (sigma_safe < 0.1 * sigma_max)
    penalty = penalty_scale * np.where(known_good, 0.5, 0.0)
    
    # Combine: base_score + exploration - penalty
    score = base_score + exploration - penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative scores (for safety)
    score = np.maximum(score, 0.0)
    
    return score
