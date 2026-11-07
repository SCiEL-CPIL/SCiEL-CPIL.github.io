"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and Probability of Improvement (PI)
    normalized by cost, with an adaptive exploration bonus.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid overflow in exp for very negative/positive z
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement: Phi(z)
    pi = norm.cdf(z_clipped)
    
    # Cost normalization: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # EI per unit cost
    ei_cost = ei / cost_safe
    
    # PI per unit cost
    pi_cost = pi / cost_safe
    
    # Adaptive exploration bonus: scale with uncertainty and inverse cost
    # Use a slightly higher weight than before to encourage more exploration
    exploration = 0.1 * sigma_safe / cost_safe
    
    # Combine: primary signal is EI/cost, secondary is PI/cost, plus exploration
    # Weighted combination: 0.7 * EI/cost + 0.3 * PI/cost + exploration
    scores = 0.7 * ei_cost + 0.3 * pi_cost + exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler score
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        simple_scores = np.maximum(f_best - mu, 0.0) / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
