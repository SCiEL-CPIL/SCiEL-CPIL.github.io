"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    For each candidate, we sample a function value from the GP posterior:
        y = mu - sigma * z, where z ~ N(0,1)
    
    The score is the expected improvement per unit cost:
        score = max(0, f_best - y) / cost
    
    This encourages exploring uncertain regions (high sigma) while being
    cost-aware (preferring cheaper evaluations).
    """
    n = len(mu)
    
    # Generate random samples from standard normal distribution
    # Using a fixed seed based on the current state would make it deterministic,
    # but we want exploration across iterations. Using np.random directly
    # ensures different samples each call.
    z = np.random.standard_normal(n)
    
    # Sampled function values: y = mu - sigma * z
    # Since we're minimizing, a lower y is better.
    # When z > 0, y < mu (potentially better)
    # When z < 0, y > mu (worse)
    y = mu - sigma * z
    
    # Improvement relative to current best (higher is better for maximization)
    improvement = np.maximum(0.0, f_best - y)
    
    # Cost-aware normalization: prefer lower cost
    cost_safe = np.maximum(cost, 1e-9)
    score = improvement / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
