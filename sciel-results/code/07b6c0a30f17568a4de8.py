# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist


def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware acquisition function for cost-aware Bayesian optimization.
    
    Combines cost-normalized Expected Improvement with a crowding penalty
    based on GP uncertainty to penalize redundant evaluations in dense clusters.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best observed value so far
        cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Diversity-aware crowding penalty:
    # We approximate "crowding" using the GP's own uncertainty structure.
    # Points with very low sigma are in well-explored regions (dense clusters).
    # We create a penalty that is high where sigma is low (crowded) and low where sigma is high (exploratory).
    # 
    # Strategy: Use a normalized version of sigma as a diversity signal.
    # Low sigma -> high penalty (crowded), High sigma -> low penalty (diverse).
    # 
    # We scale sigma to [0, 1] range based on its own distribution, then use
    # an inverse relationship to create the penalty.
    
    sigma_min = np.percentile(sigma, 1)  # Robust min
    sigma_max = np.percentile(sigma, 99)  # Robust max
    sigma_range = max(sigma_max - sigma_min, 1e-9)
    
    # Normalized sigma in [0, 1]
    sigma_norm = (sigma - sigma_min) / sigma_range
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    
    # Crowding penalty: high when sigma is low (crowded region)
    # Use a power to make the penalty more aggressive in very dense regions
    # penalty = (1 - sigma_norm)^2 gives strong penalty in low-sigma areas
    crowding_penalty = (1.0 - sigma_norm) ** 2
    
    # Scale the penalty to be comparable to ei_cost
    # We want the penalty to be meaningful but not dominate completely
    # Use the median ei_cost as a scale reference
    ei_cost_median = np.median(ei_cost)
    penalty_scale = max(ei_cost_median, 1e-9) * 0.5
    
    # Final acquisition score
    score = ei_cost - penalty_scale * crowding_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
