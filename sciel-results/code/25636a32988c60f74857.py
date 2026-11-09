# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering and hypervolume contribution.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: scalar, lowest observed objective value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of scores (higher = evaluate next).
    """
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware Pareto Front Identification
    # We want to maximize EI and minimize Cost.
    # A point is on the Pareto front if no other point has both higher/equal EI and lower/equal Cost.
    
    # Sort indices by Cost (ascending). Ties broken by EI (descending) to ensure
    # that if costs are equal, the one with higher EI is considered first (and thus dominates).
    order = np.lexsort((-ei, cost))
    
    # Vectorized Pareto check:
    # A point is Pareto optimal if its EI is strictly greater than the maximum EI 
    # seen among all points with strictly lower cost (or equal cost but processed earlier).
    # Since we sorted by Cost asc, we can just check if EI[i] > max(EI[0..i-1]).
    # Note: If costs are equal, the first one (highest EI) is kept, others are dominated.
    
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    # Cumulative maximum of EI in the sorted order
    # We need to be careful with equal costs. 
    # If cost[i] == cost[i-1], and ei[i] <= ei[i-1], it is dominated.
    # Our sort puts higher EI first for same cost. So ei[i] <= ei[i-1] is always true for duplicates.
    # Thus, cummax works perfectly: a point is Pareto if ei[i] > cummax(ei[0:i])
    
    # To handle the "strictly greater" condition for dominance:
    # If ei[i] == max_prev_ei, it is dominated (since cost is >= prev cost, and if cost == prev cost, EI is <=).
    # So we check if ei[i] > max_prev_ei.
    
    max_ei_prev = np.maximum.accumulate(ei_sorted)
    # Shift max_ei_prev by 1 to get max of previous elements
    # max_prev[i] = max(ei_sorted[0:i])
    max_prev = np.zeros_like(ei_sorted)
    if n > 1:
        max_prev[1:] = max_ei_prev[:-1]
    
    # Pareto mask in sorted order
    # A point is Pareto if its EI is strictly greater than the max EI of all previous points.
    is_pareto_sorted = ei_sorted > max_prev
    
    # Map back to original indices
    is_pareto = np.zeros(n, dtype=bool)
    is_pareto[order] = is_pareto_sorted
    
    # Calculate Hypervolume Contribution
    # Reference point: (0, C_max) where C_max is the maximum cost in the set.
    # We consider the Pareto front sorted by Cost ascending.
    # Contribution of point i is (C_{i+1} - C_i) * EI_i, where C_{last+1} = C_max.
    
    cost_max = np.max(cost)
    scores = np.zeros(n, dtype=np.float64)
    
    if np.any(is_pareto):
        # Get Pareto indices sorted by cost
        pareto_indices = np.where(is_pareto)[0]
        pareto_indices = pareto_indices[np.argsort(cost[pareto_indices])]
        
        # We can vectorize the HV calculation
        # Let P be the array of Pareto indices sorted by cost
        # C_P = cost[P], EI_P = ei[P]
        # Next costs: C_P[1:] and C_max for the last one
        # Widths: C_P[1:] - C_P[:-1], and C_max - C_P[-1]
        
        C_P = cost[pareto_indices]
        EI_P = ei[pareto_indices]
        
        # Create array of next costs
        next_costs = np.empty_like(C_P)
        if len(C_P) > 1:
            next_costs[:-1] = C_P[1:]
        next_costs[-1] = cost_max
        
        widths = next_costs - C_P
        # Ensure widths are non-negative (should be, due to sorting)
        widths = np.maximum(widths, 0.0)
        
        hv_contribs = EI_P * widths
        
        scores[pareto_indices] = hv_contribs
        
        # Optional: Add a small bonus for points that are "central" on the front?
        # Or just stick to pure HV.
        
        # Ensure finite
        scores[~np.isfinite(scores)] = 0.0
    
    # For non-Pareto points, score is 0.
    # The evaluator will pick the max score. If all are 0 (e.g., EI=0 everywhere), it picks index 0.
    # This is acceptable.
    
    # Add tiny epsilon to break ties deterministically if needed, though not strictly required
    # scores += 1e-12 * np.arange(n) # No, this biases towards high indices.
    
    return scores
# EVOLVE-BLOCK-END
