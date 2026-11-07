"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Cost-Weighted Expected Improvement (CW-EI) with an enhanced exploration bonus.
    The exploration bonus encourages visiting high-uncertainty regions, which is crucial
    for global convergence in the early stages, while the EI term drives exploitation.
    Both are normalized by the evaluation cost to prioritize cheaper evaluations.
    """
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Cost normalization: cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: Encourages exploring high uncertainty regions.
    # A higher coefficient than the previous 0.1 to ensure sufficient global search.
    exploration = 0.5 * sigma_safe
    
    # Combine EI and Exploration
    combined = ei + exploration
    
    # Normalize by cost
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Fallback to simple EI/cost
        simple_scores = ei / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
