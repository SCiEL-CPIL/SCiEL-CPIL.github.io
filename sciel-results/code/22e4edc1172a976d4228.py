"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a normalized trade-off between
    Expected Improvement (EI) and Evaluation Cost.
    
    Strategy:
    1. Compute Expected Improvement (EI) for each candidate.
    2. Normalize EI and Cost to [0, 1] ranges based on the current batch of candidates.
    3. Combine them into a score: Score = Norm_EI - lambda * Norm_Cost.
    This effectively selects points from the Pareto front of (High EI, Low Cost).
    """
    # Deadline guard (safety margin)
    start_time = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # 1. Compute Expected Improvement (EI)
    # Handle numerical stability for sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Normalize EI and Cost
    # Add small epsilon to avoid division by zero if max==min
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
        
    # 3. Combine scores
    # Lambda controls the trade-off. 
    # A value of 1.0 implies that a 10% increase in normalized cost 
    # is equivalent to a 10% decrease in normalized EI.
    # Given the tight budget, we penalize cost moderately.
    lambda_cost = 0.5
    scores = ei_norm - lambda_cost * cost_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline check
    if time.time() - start_time > 0.5:
         # Fallback: just return EI if we are running out of time (unlikely)
         return ei
         
    return scores
# EVOLVE-BLOCK-END
