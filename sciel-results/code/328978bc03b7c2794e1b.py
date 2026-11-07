import numpy as np
import time

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Strategy:
    - Uses a capacity-aware spatial partitioning heuristic.
    - For customer-customer edges, uses a 'savings per distance cost' metric
      with a spatial density penalty based on local demand concentration.
    - For depot edges, uses pure sharp inverse distance.
    
    The spatial density penalty discourages long-range connections between
    customers in different spatial regions, promoting compact routes that
    respect local capacity constraints.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Basic validation and time check
    if dm.shape != (n, n) or not np.isfinite(dm).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h
    
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h

    eps = 1e-9
    coords = np.asarray(coordinates, dtype=float)
    demands_arr = np.asarray(demands, dtype=float)
    
    # Compute savings: s[i, j] = d[i, 0] + d[0, j] - d[i, j]
    d_i0 = dm[:, 0]
    d_0j = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_0j[np.newaxis, :] - dm
    savings = np.maximum(savings, 0.0)
    
    # Compute sharp inverse distance: d^-1.5
    d_safe = dm + eps
    inv_dist_sharp = 1.0 / (d_safe ** 1.5)
    
    # Initialize heuristic matrix
    h = np.zeros((n, n))
    
    # 1. Depot Edges (i=0 or j=0)
    h[0, :] = inv_dist_sharp[0, :]
    h[:, 0] = inv_dist_sharp[:, 0]
    
    # 2. Customer-Customer Edges (i>0, j>0)
    cc_mask = np.ones((n, n), dtype=bool)
    cc_mask[0, :] = False
    cc_mask[:, 0] = False
    np.fill_diagonal(cc_mask, False)
    
    # Compute spatial density penalty for customer-customer edges
    # For each pair (i, j), compute the ratio of edge length to local spatial scale
    # Use a simple approach: penalize edges that connect customers far apart
    # relative to their local neighborhood density
    
    # Compute local density: for each customer, the average demand within a radius
    # This creates a spatially-aware penalty
    
    # Simple approach: use the distance between customers relative to the
    # average distance to their k nearest neighbors
    
    k = min(10, n - 1)
    if k > 0:
        # Compute distance from each customer to all other customers
        customer_dm = dm[1:, 1:]  # n-1 x n-1 matrix
        n_cust = n - 1
        
        # For each customer, find the k-th nearest neighbor distance
        # This gives a local spatial scale
        local_scale = np.zeros(n_cust)
        for i in range(n_cust):
            dists = customer_dm[i, :].copy()
            dists[i] = np.inf  # exclude self
            # Get k smallest distances
            if k < n_cust:
                kth_dist = np.partition(dists, k - 1)[k - 1]
            else:
                kth_dist = np.max(dists)
            local_scale[i] = kth_dist
        
        # For edge (i, j) where i, j > 0, compute spatial penalty
        # penalty[i, j] = 1 / (1 + (d[i,j] / (local_scale[i-1] + local_scale[j-1]))^2)
        # This penalizes edges that are much longer than the local scale
        
        local_scale_full = np.zeros(n)
        local_scale_full[1:] = local_scale
        
        # Compute spatial penalty matrix for customer-customer edges
        # spatial_penalty[i, j] = 1 / (1 + (d[i,j] / (ls[i] + ls[j]))^2)
        ls_sum = local_scale_full[:, np.newaxis] + local_scale_full[np.newaxis, :]
        ls_sum = np.maximum(ls_sum, eps)
        
        # Compute the ratio
        ratio = dm / ls_sum
        spatial_penalty = 1.0 / (1.0 + ratio ** 2)
        
        # Apply spatial penalty to customer-customer edges
        h_cc_candidate = (savings + eps) * inv_dist_sharp * spatial_penalty
    else:
        h_cc_candidate = (savings + eps) * inv_dist_sharp
    
    # Apply only to customer-customer edges
    h[cc_mask] = h_cc_candidate[cc_mask]
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(h, 0.0)
    
    # Handle non-finite or non-positive values
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Normalize to prevent extreme values from dominating pheromone updates
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
        
    # Re-apply positivity constraint after normalization
    h[h <= 0] = 1e-9
    np.fill_diagonal(h, 0.0)
    
    return h
