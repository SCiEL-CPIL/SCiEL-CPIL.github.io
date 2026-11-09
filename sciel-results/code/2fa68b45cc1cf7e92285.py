"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 280  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Lagrangian relaxation approach: score = f_best - mu - lambda * cost,
    where lambda is dynamically tuned to balance exploration vs cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound for minimization
    beta = 1.0
    lcb = mu - beta * sigma
    
    # Cost-aware Lagrangian: score = EI + alpha * sigma - lambda * cost
    # lambda is tuned so that the average cost of selected points is balanced
    
    # Compute a baseline score without cost penalty
    base_score = ei + 0.5 * sigma
    
    # Dynamically tune lambda based on the scale of scores vs cost
    # We want lambda such that cost penalty is meaningful relative to score scale
    score_scale = np.std(base_score)
    if score_scale < 1e-9:
        score_scale = 1.0
    
    cost_scale = np.std(cost)
    if cost_scale < 1e-9:
        cost_scale = 1.0
    
    # lambda = score_scale / cost_scale * penalty_factor
    # penalty_factor controls how strongly we penalize cost
    # Use a moderate penalty that encourages cost-awareness without dominating
    penalty_factor = 0.3
    lam = penalty_factor * score_scale / cost_scale
    
    # Final score: combine EI, exploration, and cost penalty
    scores = base_score - lam * cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check - if close to timeout, return quickly
    if not _elapsed_ok():
        return ei / cost
    
    return scores
# EVOLVE-BLOCK-END
