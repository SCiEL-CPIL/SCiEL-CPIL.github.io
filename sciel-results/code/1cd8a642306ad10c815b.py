# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
DEADLINE_LIMIT = 330.0
SAFETY_MARGIN = 35.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard EI for minimization with exploitation bonus xi."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using EI with exponential cost penalty.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement
    # Small xi for slight exploitation bias
    xi = 0.005 * abs(f_best)
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    
    # Exponential cost penalty: exp(-lambda * cost)
    # lambda dynamically scaled by optimization progress
    # When f_best is large (early), lambda is small -> less cost sensitivity
    # When f_best is small (late), lambda is larger -> more cost sensitivity
    # This creates a soft barrier that favors low-cost candidates more as we converge
    f_best_abs = abs(f_best)
    # lambda scales inversely with f_best: when f_best ~ 100, lambda ~ 0.01
    # when f_best ~ 1, lambda ~ 0.1
    # when f_best ~ 0.1, lambda ~ 1.0
    lambda_cost = 1.0 / (1.0 + f_best_abs)
    
    # Apply exponential penalty to EI
    # cost is in [1, 2] for this problem
    # exp(-lambda * cost) will be in [exp(-2*lambda), exp(-lambda)]
    cost_penalty = np.exp(-lambda_cost * cost)
    
    # Final score: EI * cost_penalty
    score = ei * cost_penalty
    
    # Add a small UCB component for exploration, also cost-penalized
    # UCB for minimization: mu - beta * sigma
    # Use adaptive beta: more exploration when f_best is large
    beta = 2.0 / (1.0 + 0.1 * f_best_abs)
    ucb = mu - beta * sigma
    
    # Normalize UCB to be comparable to EI scale
    # EI has units of f, UCB also has units of f
    # Scale UCB contribution
    ucb_weight = 0.1  # Small weight for UCB
    ucb_contribution = ucb_weight * np.maximum(f_best - ucb, 0.0) * cost_penalty
    
    score = score + ucb_contribution
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _remaining_budget() < 5.0:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-12)
        # Simple EI with cost penalty
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        lambda_cost = 1.0 / (1.0 + abs(f_best))
        cost_penalty = np.exp(-lambda_cost * cost)
        score = ei * cost_penalty
        return np.where(np.isfinite(score), score, 0.0)
    
    return _acquisition(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
