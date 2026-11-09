"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _compute_cost(x):
    """Compute cost for points x of shape (n, 4)."""
    x_scaled = x / 10.0  # scale to [0, 1]
    return 1.0 + np.mean(x_scaled, axis=1)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Probability of Improvement (PoI) and
    Expected Improvement (EI), weighted by cost, with adaptive
    exploration based on uncertainty.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (PoI)
    poi = norm.cdf(z)
    
    # Compute a measure of how "improved" we are - helps with exploration
    # When f_best is close to mu, we want more exploration
    # When f_best is far from mu (mu << f_best), we want exploitation
    
    # Adaptive mixing: use more PoI when uncertainty is high relative to improvement
    # This encourages exploration in uncertain regions
    
    # Normalize EI and PoI to [0, 1] roughly for mixing
    # EI can be negative if mu > f_best (unlikely but possible)
    ei = np.maximum(ei, 0.0)
    
    # Weight based on cost: divide by cost to get "improvement per unit cost"
    ei_cost = ei / cost
    poi_cost = poi / cost
    
    # Mix EI and PoI with a fixed weight
    # PoI is more exploratory, EI is more exploitative
    # Use 0.5/0.5 mix as a starting point
    score = 0.5 * ei_cost + 0.5 * poi_cost
    
    # Add a small bonus for high uncertainty to encourage exploration
    # This helps escape local optima in multimodal functions
    uncertainty_bonus = sigma / (sigma + np.std(mu) + 1e-8)
    score += 0.1 * uncertainty_bonus / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
