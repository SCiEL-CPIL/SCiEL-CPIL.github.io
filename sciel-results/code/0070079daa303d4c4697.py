"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 285  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
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
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # Lagrangian-based cost-aware acquisition:
    # Maximize EI(x) - lambda * cost(x)
    # where lambda is the shadow price of budget.
    #
    # Key insight: lambda should reflect the marginal value of remaining budget.
    # Early on (high uncertainty, little data), we can afford to explore more.
    # Late on (tight budget), we should be more selective.
    
    avg_cost = np.mean(cost)
    # Estimate how many more evaluations we can afford
    # This is approximate; the actual remaining budget is not passed in,
    # but we can use the average cost as a proxy for budget pressure.
    
    # Adaptive lambda: higher when EI is small relative to cost (budget-tight regime)
    # Lower when EI is large (plenty of improvement to justify cost)
    ei_mean = np.mean(ei)
    if ei_mean > 1e-12:
        # lambda scales with cost efficiency: if average EI per cost is low,
        # we need a higher penalty to be selective
        lambda_base = 0.5 * ei_mean / max(avg_cost, 1e-9)
    else:
        lambda_base = 0.1
    
    # Add a small exploration bonus via UCB term scaled by uncertainty
    avg_sigma = np.mean(sigma)
    std_sigma = max(np.std(sigma), 1e-9)
    
    # Exploration bonus: points with high uncertainty get a small boost
    # This helps escape local optima in the acquisition landscape
    ucb_bonus = 0.15 * sigma / (avg_sigma + 1e-9)
    
    # Final score: EI - lambda * cost + exploration_bonus
    scores = ei - lambda_base * cost + ucb_bonus * ei
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're close to timeout, return quickly
    if not _elapsed_ok():
        return ei / cost
    
    return scores
# EVOLVE-BLOCK-END
