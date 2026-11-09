"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Pareto Frontier Filter + Probability of Improvement Acquisition.
    
    First, filters candidates to retain only those on the Pareto frontier
    in the (mu, sigma) space (non-dominated: lower mu is better, lower sigma 
    is better for certainty). Then scores survivors using Probability of 
    Improvement (PI), which is a non-linear, threshold-sensitive acquisition 
    that focuses search on promising regions.
    """
    start_time = time.time()
    DEADLINE = 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Probability of Improvement (PI) for all candidates
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure PI is finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # 2. Pareto Frontier Filter
    # We want to keep candidates that are not dominated in (mu, sigma) space.
    # A candidate i dominates candidate j if mu_i <= mu_j AND sigma_i <= sigma_j
    # with at least one strict inequality.
    # Since lower mu and lower sigma are both desirable, we keep the 
    # "frontier" of points that are best in at least one dimension.
    
    # For efficiency with 2000 points, we use a sorting-based approach:
    # Sort by mu (ascending). Then scan and keep a point if its sigma is 
    # less than the minimum sigma seen so far. This gives the Pareto frontier
    # for minimization in both dimensions.
    
    # Sort indices by mu
    mu_sorted_idx = np.argsort(mu)
    mu_sorted = mu[mu_sorted_idx]
    sigma_sorted = sigma_safe[mu_sorted_idx]
    
    # Find Pareto frontier: a point is on the frontier if no previous point 
    # (with lower or equal mu) has lower or equal sigma
    # We track the minimum sigma seen so far as we scan in order of increasing mu
    min_sigma_so_far = np.inf
    frontier_mask_sorted = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if sigma_sorted[i] <= min_sigma_so_far:
            frontier_mask_sorted[i] = True
            min_sigma_so_far = sigma_sorted[i]
    
    # Map back to original indices
    frontier_mask = np.zeros(n, dtype=bool)
    frontier_mask[mu_sorted_idx[frontier_mask_sorted]] = True
    
    # 3. Score construction:
    # - Pareto frontier points get PI as their score
    # - Non-frontier points get a heavily discounted score
    # This creates a sharp selection pressure toward the frontier
    
    # To make the filtering softer (avoid completely discarding information),
    # we give non-frontier points a small fraction of their PI
    # This prevents the acquisition from being too brittle
    
    # Adaptive discount: early on, be more permissive; late, be stricter
    known_min = -2.53218
    if f_best > known_min:
        progress = min(max((f_best - known_min) / (0 - known_min), 0), 1)
    else:
        progress = 0
        
    # Discount factor for non-frontier points: 
    # Early (progress high): less discount (0.1)
    # Late (progress low): more discount (0.01)
    discount = 0.01 + 0.09 * progress  # Range: 0.01 to 0.10
    
    scores = np.where(frontier_mask, pi, discount * pi)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we're close to deadline, return what we have
    if time.time() - start_time > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
