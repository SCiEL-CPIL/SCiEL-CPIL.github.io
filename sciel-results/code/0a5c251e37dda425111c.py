"""Rewritten acquisition for cost-aware Bayesian optimization of Rastrigin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining:
    - Expected Improvement (EI) with a small exploration offset
    - Uncertainty bonus to escape local minima of Rastrigin
    - Cost normalization (cheaper points get higher score)
    
    mu: array of GP mean predictions, shape (2000,)
    sigma: array of GP std predictions, shape (2000,)
    f_best: scalar, best observed function value so far
    cost: array of evaluation costs, shape (2000,)
    Returns: array of acquisition scores, shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement with small xi for slight exploration
    xi = 0.005
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # EI formula
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: encourages exploration in high-sigma regions
    # This is important for Rastrigin which has many local minima
    # Use a saturating function to prevent unbounded growth
    uncertainty_bonus = 0.03 * sigma_safe / (1.0 + sigma_safe)
    
    # Cost-aware: divide by cost so cheaper evaluations are favored
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    cost_normalized_ei = ei / cost_safe
    
    # Combine: cost-normalized EI + uncertainty bonus
    # The uncertainty bonus is NOT divided by cost because we want to
    # always explore uncertain regions regardless of cost, but with a
    # small weight to not dominate the EI signal
    score = cost_normalized_ei + uncertainty_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
