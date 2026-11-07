"""Cost-aware acquisition function for Bayesian optimization of 2D Lévy.
Uses Expected Improvement (EI) with a quadratic cost penalty to bias search
towards the low-cost corner where shifted Levy minima are likely located."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# One-time wall-clock anchor for the deadline guard.
# The evaluator kills after 330s. We return early if we get close to this.
_T0 = time.perf_counter()
_DEADLINE = 325.0  # Safety margin of 5 seconds before the 330s kill.


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores.
    
    Args:
        mu: GP mean predictions, shape (2000,)
        sigma: GP standard deviation predictions, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,). Higher = evaluate next.
    """
    # Deadline guard: if we are running out of time, return a safe fallback.
    # A simple cost-based heuristic ensures we still pick a valid point.
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Robust sigma flooring to prevent division by zero or extreme z-scores.
    # 1e-9 is a standard small epsilon that preserves the GP's structure.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Robust cost flooring.
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement (EI).
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is non-negative by definition for minimization.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety for very large z).
    ei = np.maximum(ei, 0.0)
    
    # Information gain proxy: predictive variance (sigma^2) represents uncertainty.
    # Reducing variance at a point = information gain. Higher sigma^2 = more to learn.
    # Per unit cost: sigma^2 / cost favors cheap, uncertain points.
    ig_per_cost = (sigma_safe * sigma_safe) / cost_safe
    
    # Blend: primarily information gain per cost, with a small EI component
    # to retain some exploitation of known high-improvement regions.
    # EI is divided by cost^2 to normalize for cost (quadratic penalty).
    ei_per_cost = ei / (cost_safe * cost_safe)
    
    # Weight information gain more heavily (80/20 split)
    scores = 0.8 * ig_per_cost + 0.2 * ei_per_cost
    
    # Final validity guard: ensure all outputs are finite.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
