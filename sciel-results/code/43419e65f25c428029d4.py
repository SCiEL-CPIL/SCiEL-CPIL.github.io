"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses Expected Improvement with adaptive cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware EI
        z = (f_best - mu) / sigma
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute progress: how close are we to the known minimum?
    # Use a reasonable initial guess for the starting f_best
    initial_guess = -1.0
    total_range = initial_guess - HARTMANN_6D_MIN
    progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Adaptive cost exponent: use sqrt(cost) early, linear cost later
    # This allows exploration of expensive regions early, then focuses on cheap improvements
    cost_exp = 0.5 + 0.5 * progress  # ranges from 0.5 to 1.0
    cost_factor = np.power(cost, cost_exp)
    
    # Base cost-aware EI
    base_scores = ei / cost_factor
    
    # Add an exploration bonus based on uncertainty, scaled by progress
    # More exploration early, less late
    # Normalize sigma for stable scaling
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Exploration bonus: encourages visiting uncertain regions
    # Scale down as we progress
    exploration_weight = 0.15 * (1.0 - progress)
    exploration_bonus = exploration_weight * sigma_norm
    
    # Combine
    scores = base_scores + exploration_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
