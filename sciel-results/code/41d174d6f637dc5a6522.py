# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a multiplicative feasibility gate
    based on the average cost of promising candidates.
    
    The score is Expected Improvement (EI) only if the predicted cost is below
    the mean cost of candidates with positive expected improvement. This creates
    a dynamic budget-aware gate that adapts to the GP's belief about where
    improvements are likely to be found, favoring cheaper options among the
    promising set.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Identify promising candidates (those with significant expected improvement)
    # Use a small epsilon to filter out negligible improvements
    epsilon = 1e-6
    promising_mask = ei > epsilon
    
    if np.any(promising_mask):
        # Calculate the average cost of the promising candidates
        # This serves as the dynamic threshold
        mean_cost_promising = np.mean(cost[promising_mask])
        
        # Gate: only allow EI if cost is below this average cost
        # This forces the selection of the cheaper half (on average) of the
        # promising points.
        affordable = cost <= mean_cost_promising
        
        scores = np.where(affordable, ei, 0.0)
    else:
        # Fallback: if no promising points, just use EI (or return zeros? 
        # Returning EI is safer to maintain exploration if f_best is stuck)
        scores = ei.copy()
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
