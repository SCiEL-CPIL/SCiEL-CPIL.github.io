"""Rewritten cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: ensure we finish well before 300s
_DEADLINE = time.time() + 280.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining normalized EI/cost with exploration.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: improvement per unit cost
    ei_score = ei / cost_safe
    
    # Upper Confidence Bound with cost awareness
    # UCB = beta * sigma / cost
    beta = 0.5
    ucb_score = beta * sigma / cost_safe
    
    # Additional exploration: prefer points far from f_best in relative terms
    # This helps when f_best is not very good yet
    rel_improvement = (f_best - mu) / (np.abs(f_best) + 1e-6)
    rel_improvement = np.clip(rel_improvement, -10, 10)
    rel_score = rel_improvement / cost_safe
    
    # Normalize each component to zero-mean, unit-variance
    def _normalize(arr):
        m = np.mean(arr)
        s = np.std(arr) + 1e-8
        return (arr - m) / s
    
    ei_norm = _normalize(ei_score)
    ucb_norm = _normalize(ucb_score)
    rel_norm = _normalize(rel_score)
    
    # Combine components
    # Weight EI heavily (exploitation), moderate UCB (exploration), small relative improvement
    combined = 0.65 * ei_norm + 0.25 * ucb_norm + 0.10 * rel_norm
    
    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
