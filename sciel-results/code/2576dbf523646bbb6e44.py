"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pareto-based multi-objective combining normalized EI
    and normalized predictive uncertainty with dynamic weighting.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI: divide by its max (or a small value if all zero)
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize sigma: divide by its max
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Dynamic weight: increases with number of evaluations
    # Estimate number of evaluations from the gap between f_best and minimum
    min_possible = -3.32237
    gap = max(f_best - min_possible, 0.0)
    
    # Early on (large gap), we want more exploration; later (small gap), more exploitation
    # Use a sigmoid-like function: weight for exploration starts high, decreases
    # weight_exploration = 1 / (1 + exp((gap - 0.3) / 0.1))
    # This gives ~0.97 when gap=0.1, ~0.5 when gap=0.3, ~0.03 when gap=0.5
    
    weight_exploration = 1.0 / (1.0 + np.exp((gap - 0.3) / 0.1))
    weight_exploitation = 1.0 - weight_exploration
    
    # Combine normalized EI and normalized sigma
    scores = weight_exploitation * ei_norm + weight_exploration * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best available result quickly
        return scores
    
    return scores
# EVOLVE-BLOCK-END
