import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Clamp small negative values to 0.
    ei = np.maximum(ei, 0.0)
    
    # Initialize scores to 0
    scores = np.zeros_like(ei)
    
    # Identify Pareto Optimal Candidates
    # Objective: Maximize EI, Minimize Cost
    # Sort indices by Cost ascending
    sort_idx = np.argsort(cost)
    
    pareto_indices = []
    max_ei_so_far = -np.inf
    
    # Iterate through sorted candidates
    # A candidate is Pareto optimal if its EI is strictly greater than the max EI seen so far (with lower or equal cost)
    # Note: If costs are equal, we only keep the one with the highest EI.
    # The simple "running max" works if we process in order of increasing cost.
    # If cost[i] == cost[j] and i comes before j:
    # If ei[j] > ei[i], then j dominates i.
    # The running max approach:
    # We update max_ei_so_far.
    # If current ei > max_ei_so_far, it's Pareto.
    # This correctly handles ties in cost by only keeping the highest EI for that cost bucket effectively,
    # because if a later point has same cost and higher EI, it will be Pareto, and the earlier one (lower EI) was already marked Pareto?
    # Wait.
    # If cost is same, and EI increases:
    # Point 1: C=1, EI=1. Max=-inf. 1 > -inf. Pareto. Max=1.
    # Point 2: C=1, EI=2. Max=1. 2 > 1. Pareto. Max=2.
    # Both are marked Pareto. But Point 1 is dominated by Point 2 (same cost, lower EI).
    # So we need to be careful.
    
    # Refined Pareto Filter:
    # Sort by Cost ascending.
    # We want to keep points that are not dominated by any point with lower cost.
    # And among points with same cost, only the max EI is non-dominated by other points with same cost.
    
    # Let's use a different approach for robustness:
    # 1. Sort by Cost ascending.
    # 2. Iterate. Keep track of max_ei_seen.
    # 3. If current EI > max_ei_seen, it is a "candidate" for Pareto.
    # 4. However, if there are multiple points with the same cost, only the one with the highest EI among those with that cost (that are > max_ei_seen) should be kept?
    # Actually, if C_i == C_j and EI_i < EI_j, i is dominated by j.
    # If we process i then j:
    # i: EI_i > max_ei_so_far. Mark i. Update max_ei_so_far = EI_i.
    # j: EI_j > max_ei_so_far (since EI_j > EI_i). Mark j. Update max_ei_so_far = EI_j.
    # Both marked. But i is dominated.
    
    # Correction:
    # We should only mark a point as Pareto if it is NOT dominated by any OTHER point.
    # Since we sort by Cost, any point that dominates it must have Cost <= C_i.
    # If Cost < C_i, it must have EI >= EI_i.
    # If Cost == C_i, it must have EI >= EI_i.
    
    # Efficient O(N log N) or O(N) after sort:
    # Sort by Cost ascending.
    # For groups of same Cost, keep only the max EI.
    # Then, on the unique cost groups, apply the running max logic.
    
    # 1. Get unique costs and max EI for each
    # Since N=2000, we can do this simply.
    
    # Sort indices by cost
    sorted_indices = np.argsort(cost)
    sorted_costs = cost[sorted_indices]
    sorted_eis = ei[sorted_indices]
    
    # Find Pareto indices in the sorted array
    # We traverse the sorted array.
    # We maintain max_ei_seen.
    # We also need to handle ties in cost.
    
    pareto_mask_sorted = np.zeros(len(sorted_indices), dtype=bool)
    
    current_max_ei = -np.inf
    i = 0
    n = len(sorted_indices)
    
    while i < n:
        # Find all indices with the same cost
        current_cost = sorted_costs[i]
        j = i
        while j < n and sorted_costs[j] == current_cost:
            j += 1
        # In the range [i, j), find the max EI
        block_eis = sorted_eis[i:j]
        max_block_ei = np.max(block_eis)
        
        if max_block_ei > current_max_ei:
            # The points in this block that have EI == max_block_ei are Pareto candidates
            # Actually, if max_block_ei > current_max_ei, then the best point in this block is Pareto.
            # Points in this block with lower EI are dominated by the best point in this block (same cost).
            # So we mark only the indices in [i, j) where EI == max_block_ei.
            # But wait, if max_block_ei == current_max_ei, then this block is dominated by previous blocks?
            # If max_block_ei <= current_max_ei, no point in this block is Pareto (dominated by a point with lower cost and higher/equal EI).
            
            # So:
            if max_block_ei > current_max_ei:
                # Mark indices in [i, j) where EI is max_block_ei
                for k in range(i, j):
                    if sorted_eis[k] == max_block_ei:
                        pareto_mask_sorted[k] = True
                current_max_ei = max_block_ei
        # else:
        # All points in this block are dominated by previous points.
        
        i = j
    
    # Map back to original indices
    pareto_indices_original = sorted_indices[pareto_mask_sorted]
    
    if len(pareto_indices_original) == 0:
        # Fallback: if no Pareto points (should not happen if EI > 0 exists), return EI / cost
        # Or just return EI
        return ei / (cost + 1e-9)
    
    # Calculate Hypervolume Contribution
    # We need the Pareto points sorted by Cost ascending.
    # They are already in sorted_indices order, filtered by mask.
    # Extract their costs and EIs.
    
    p_indices = pareto_indices_original
    p_costs = cost[p_indices]
    p_eis = ei[p_indices]
    
    # Sort them by cost again to be sure (they should be, but good practice)
    # p_indices is derived from sorted_indices which is sorted by cost.
    # So p_costs is non-decreasing.
    
    # Reference Cost: Max cost in the entire set? Or max cost in Pareto set?
    # The directive says "relative to ... origin".
    # Let's use C_ref = max(cost) (global max) to maximize the "width" for low cost points.
    C_ref = np.max(cost)
    I_ref = 0.0
    
    # Calculate contributions
    # H_i = (C_{i+1} - C_i) * (I_i - I_ref)
    # For the last point, C_{m+1} = C_ref.
    
    m = len(p_indices)
    hv_contrib = np.zeros(m)
    
    if m > 0:
        for k in range(m):
            if k < m - 1:
                c_next = p_costs[k+1]
            else:
                c_next = C_ref
            
            # Ensure width is positive
            width = c_next - p_costs[k]
            if width < 0:
                width = 0.0
                
            height = p_eis[k] - I_ref
            if height < 0:
                height = 0.0
                
            hv_contrib[k] = width * height
    
    # Assign scores
    # The score is the HV contribution.
    # To ensure "higher = evaluate next", this is correct.
    
    scores[p_indices] = hv_contrib
    
    # Optional: Add a small epsilon to non-zero scores to break ties or ensure positivity?
    # The task requires finite scores. 0 is finite.
    # However, if all HV contributions are 0 (e.g., all EI are 0), scores are 0.
    # This is acceptable.
    
    return scores
