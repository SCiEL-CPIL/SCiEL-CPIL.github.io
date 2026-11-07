"""Cost-aware acquisition function for Bayesian optimization of 2D Lévy.
Uses Cost-Adjusted Upper Confidence Bound (CA-UCB) to balance exploration
and exploitation while accounting for evaluation costs.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# One-time wall-clock anchor for the deadline guard.
_T0 = time.perf_counter()
_DEADLINE = 325.0  # Safety margin of 5 seconds before the 330s kill.

# Exploration-exploitation trade-off parameter for UCB.
# Higher beta encourages more exploration.
_BETA = 2.5


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores using Cost-Adjusted UCB.
    
    Args:
        mu: GP mean predictions, shape (2000,)
        sigma: GP standard deviation predictions, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,). Higher = evaluate next.
    """
    # Deadline guard
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Robust sigma flooring
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Robust cost flooring
    cost_safe = np.maximum(cost, 1e-9)

    # UCB for minimization:
    # We want to find points with low predicted value.
    # The "optimistic" value is mu - beta * sigma.
    # We want to MAXIMIZE the acquisition score.
    # So we maximize: -(mu - beta * sigma) = -mu + beta * sigma.
    
    # However, pure UCB can be too exploratory.
    # A common refinement is to cap the exploration term or use EI.
    # Let's stick to CA-UCB for now, as it's a distinct family from EI.
    
    ucb_values = -mu + _BETA * sigma_safe
    
    # Cost-aware normalization
    scores = ucb_values / cost_safe
    
    # Final validity guard
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
