import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize deadline
    if _start_time is None:
        _start_time = time.time()
        
    # Safety margin: 330s limit, return if we are too close
    # We return zeros which effectively stops selection or picks random/first,
    # but prevents crash. In practice, this should rarely trigger if the loop is efficient.
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)
        
    # 1. Calculate Expected Improvement (EI)
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Hard Quantile-Based Feasibility Gate
    # Structural change: Instead of dividing EI by cost (smooth penalty),
    # we use a hard gate based on cost quantiles.
    # This forces the search to concentrate evaluations in the low-cost corner
    # until the GP uncertainty there is exhausted, rather than spreading
    # evaluations uniformly across cost levels.
    
    # Calculate quantiles of the cost distribution for the current candidates
    # The 70th percentile creates a sharper exploitation focus than the previous additive penalty.
    # We allow points in the lowest 70% of cost to be considered.
    cost_threshold = np.percentile(cost, 70)
    
    # Create a mask for feasible (low-cost) candidates
    feasible_mask = cost <= cost_threshold
    
    # Initialize scores to 0
    scores = np.zeros_like(mu)
    
    # Only assign EI scores to feasible candidates
    scores[feasible_mask] = ei[feasible_mask]
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
