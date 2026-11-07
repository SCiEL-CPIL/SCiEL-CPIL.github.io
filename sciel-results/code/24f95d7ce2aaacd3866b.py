"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for Bayesian optimization.
    
    This approach scores candidates by their marginal contribution to the
    hypervolume of the Pareto front of (mu, sigma) points.
    
    1. Normalize mu and sigma to [0, 1].
    2. Identify the non-dominated (Pareto) front.
    3. Compute the hypervolume contribution of each Pareto point using
       a standard 2D decomposition.
    4. Return the contributions as scores (higher is better).
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Handle edge cases where variance is zero
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    if mu_max == mu_min:
        # All mu are identical, pick lowest sigma
        return -sigma
    if sigma_max == sigma_min:
        # All sigma are identical, pick lowest mu
        return -mu
        
    # Normalize to [0, 1]
    # We want lower values to be "better", so in the normalized space for HV,
    # we map the best (min) to 0 and worst (max) to 1.
    # The reference point for minimization HV is (1, 1).
    mu_norm = (mu - mu_min) / (mu_max - mu_min)
    sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    
    # Ensure values are strictly within [0, 1] due to floating point
    mu_norm = np.clip(mu_norm, 0.0, 1.0)
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    
    # Find Pareto front
    # A point is non-dominated if no other point has mu <= mu_i and sigma <= sigma_i
    # with at least one strict inequality.
    
    # Sort by mu_norm ascending
    sorted_idx = np.argsort(mu_norm)
    sorted_mu = mu_norm[sorted_idx]
    sorted_sigma = sigma_norm[sorted_idx]
    
    # Find indices in the sorted array that are on the Pareto front
    # In sorted order (mu ascending), a point is on the Pareto front if its sigma
    # is strictly lower than the minimum sigma seen so far in the sorted list.
    # Note: Since mu is sorted ascending, we look for decreasing sigma.
    
    pareto_mask_sorted = np.zeros(n, dtype=bool)
    min_sigma_so_far = np.inf
    for i in range(n):
        if sorted_sigma[i] < min_sigma_so_far:
            pareto_mask_sorted[i] = True
            min_sigma_so_far = sorted_sigma[i]
            
    # Map back to original indices
    pareto_mask = np.zeros(n, dtype=bool)
    pareto_mask[sorted_idx] = pareto_mask_sorted
    
    # Get Pareto points
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) == 0:
        # Fallback should not happen if n>0
        return np.zeros(n)
        
    pareto_mu = mu_norm[pareto_indices]
    pareto_sigma = sigma_norm[pareto_indices]
    
    # Sort Pareto points by mu_norm ascending for HV calculation
    # They are already in an order derived from global sort, but let's be sure
    p_order = np.argsort(pareto_mu)
    p_mu = pareto_mu[p_order]
    p_sigma = pareto_sigma[p_order]
    
    # Calculate Hypervolume Contribution
    # Reference point is (1, 1)
    # Decomposition:
    # For i in 0..k-2: contribution = (p_mu[i+1] - p_mu[i]) * (1 - p_sigma[i])
    # For i = k-1: contribution = (1 - p_mu[k-1]) * (1 - p_sigma[k-1])
    
    k = len(p_mu)
    contributions = np.zeros(k)
    
    for i in range(k):
        if i < k - 1:
            width = p_mu[i + 1] - p_mu[i]
        else:
            width = 1.0 - p_mu[i]
            
        height = 1.0 - p_sigma[i]
        
        contributions[i] = width * height
    
    # Assign scores to original indices
    scores = np.zeros(n)
    for i, idx in enumerate(pareto_indices):
        # We need to map the contribution back to the correct Pareto point
        # The `contributions` array is aligned with `p_order` which is sorted `pareto_indices`
        # So `contributions[i]` corresponds to `pareto_indices[p_order[i]]`?
        # No, `p_order` is indices into `pareto_indices`?
        # Let's trace:
        # `pareto_indices` is the list of original indices.
        # `p_mu` is `mu_norm[pareto_indices]` sorted.
        # `p_order` is the argsort of `pareto_mu`.
        # So `p_mu[j]` is the j-th smallest mu among Pareto points.
        # The original index corresponding to `p_mu[j]` is `pareto_indices[p_order[j]]`.
        
        orig_idx = pareto_indices[p_order[i]]
        scores[orig_idx] = contributions[i]
        
    return scores
# EVOLVE-BLOCK-END
