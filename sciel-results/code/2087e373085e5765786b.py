"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: probability of improvement weighted by inverse cost,
    with a small exploration bonus to avoid getting stuck.
    
    The base score is PI/cost, which strictly prioritizes high-confidence
    exploitation of low-mean regions. We add a small UCB-style exploration
    term to ensure the search doesn't collapse prematurely.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: float, best observed function value
        cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of improvement: P(f(x) < f_best)
    # For a Gaussian N(mu, sigma^2), P(f < f_best) = Phi((f_best - mu) / sigma)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Clip PI to avoid numerical issues
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware PI: prioritize high-confidence improvements per unit cost
    score = pi / cost_safe
    
    # Add a small exploration bonus: prefer points with higher uncertainty
    # This prevents the search from collapsing to a single point too early
    # The bonus is scaled down so exploitation dominates
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    # Combine: exploitation + small exploration
    total_score = score + exploration_bonus
    
    # Ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
