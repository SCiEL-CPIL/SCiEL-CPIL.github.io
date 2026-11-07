"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _elapsed_ok():
    """Check if we're still within time budget."""
    return (time.time() - _start_time) < _TIME_LIMIT


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement per unit cost."""
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    return ei


def _ucb(mu, sigma, f_best, cost, kappa=2.0):
    """Compute UCB score per unit cost."""
    # For minimization, we want low values, so UCB for minimization is:
    # Score = -(mu - kappa * sigma) / cost
    # Higher score = more promising (lower expected value with uncertainty bonus)
    ucb_value = -(mu - kappa * sigma)
    return ucb_value


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # Safety: ensure cost is positive and finite
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Safety: ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Compute multiple acquisition criteria and combine them
    # 1. Expected Improvement per unit cost
    ei = _expected_improvement(mu, sigma, f_best)
    ei_score = ei / cost
    
    # 2. UCB per unit cost (for minimization)
    ucb_score = _ucb(mu, sigma, f_best, cost, kappa=2.0) / cost
    
    # 3. Probability of Improvement per unit cost
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    pi_score = pi / cost
    
    # 4. GP mean per unit cost (greedy exploitation)
    mean_score = -mu / cost  # Negate because we want to minimize
    
    # Combine scores with weights
    # Weight EI more heavily as it balances exploration and exploitation
    # Add a small weight to UCB for exploration
    scores = 0.5 * ei_score + 0.3 * ucb_score + 0.1 * pi_score + 0.1 * mean_score
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalize to have reasonable range
    # This helps with numerical stability
    scores_max = np.max(np.abs(scores))
    if scores_max > 1e-12:
        scores = scores / scores_max
    
    return scores
# EVOLVE-BLOCK-END
