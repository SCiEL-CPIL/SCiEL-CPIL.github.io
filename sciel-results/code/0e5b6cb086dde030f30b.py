"""
Savings-based heuristic for ACO CVRP: uses Clarke-Wright savings
to identify high-value edges for route construction, enhanced with
demand-efficiency weighting.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Approach:
      1. Compute Clarke-Wright savings for all customer-customer pairs.
      2. For depot edges, use inverse distance.
      3. Boost customer-customer edges based on demand efficiency (grouping heavy loads).
      4. Ensure positive values and finite results.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Initialize heuristic with inverse distance (standard)
    heuristic = 1.0 / dist_safe
    
    # Compute savings for customer-customer edges
    # Savings S_ij = d_0i + d_0j - d_ij
    # We only apply this to i, j != 0
    
    if n > 1:
        # Extract depot row/col
        d0 = distance_matrix[0, :]
        
        # Compute savings matrix for all pairs
        # S_ij = d0[i] + d0[j] - d_ij
        savings = d0[:, None] + d0[None, :] - distance_matrix
        
        # Only consider positive savings (triangle inequality usually holds, but just in case)
        savings = np.maximum(savings, 0.0)
        
        # Mask out depot edges (row 0 and col 0) because savings logic doesn't apply directly
        # For depot edges, we keep the inverse distance
        depot_mask = np.zeros((n, n), dtype=bool)
        depot_mask[0, :] = True
        depot_mask[:, 0] = True
        
        # For customer-customer edges, combine savings with demand efficiency
        # Demand efficiency: higher if demands[i] + demands[j] is a larger fraction of capacity
        # This encourages grouping customers that fill the truck well
        demand_sum = demands[:, None] + demands[None, :]
        
        # Weight savings by demand efficiency: 
        # Edges that save distance AND fill the truck are more valuable.
        # We use a linear weight: 1 + (d_i + d_j) / capacity
        demand_weight = 1.0 + (demand_sum / capacity)
        
        # Combine savings and demand weight
        # This creates a "weighted savings" metric
        weighted_savings = savings * demand_weight
        
        # Normalize the weighted savings to be in a comparable range (0 to ~1)
        # Using max is sensitive to outliers, but let's try max first. 
        # If max is 0, set to 1 to avoid div by zero.
        max_ws = np.max(weighted_savings[~depot_mask]) if np.any(~depot_mask) else 1.0
        if max_ws < 1e-9:
            max_ws = 1.0
            
        norm_ws = weighted_savings / max_ws
        
        # Combine with inverse distance
        # heuristic_cc = (1/d_ij) * (1 + norm_ws)
        # The term (1 + norm_ws) ensures that even if savings are 0, we have the base 1/d_ij.
        # If savings are high, we boost it significantly.
        
        cc_boost = (1.0 + norm_ws)
        
        # Apply to non-depot edges
        heuristic[~depot_mask] = (1.0 / dist_safe[~depot_mask]) * cc_boost[~depot_mask]
        
        # Depot edges remain 1/d_ij

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
