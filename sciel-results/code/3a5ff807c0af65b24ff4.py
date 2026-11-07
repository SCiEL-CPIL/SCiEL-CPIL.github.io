import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Initialize with true distances
    d_out = edge_distance.copy()
    
    # Find max distance to scale penalties
    max_d = np.max(d_out)
    if max_d == 0:
        return d_out

    # 1. Build tour mask
    tour_mask = np.zeros((n, n), dtype=float)
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                tour_mask[u, v] = 1.0
                tour_mask[v, u] = 1.0
    except Exception:
        return edge_distance.copy()

    # Geometry-aware inverse distance weighting
    # Use squared inverse distance (1/d^2) for a stronger repulsive force on short edges
    # This creates a steeper gradient for escaping local optima dominated by short edges
    safe_dist = np.where(edge_distance > 1e-9, edge_distance, 1.0)
    inv_d_sq = 1.0 / (safe_dist * safe_dist)
    max_inv_d_sq = np.max(inv_d_sq)
    if max_inv_d_sq > 0:
        inv_d_sq_norm = inv_d_sq / max_inv_d_sq
    else:
        inv_d_sq_norm = np.ones_like(edge_distance)
    
    # Zero out diagonal for penalty calculation safety
    np.fill_diagonal(inv_d_sq_norm, 0.0)

    # 2. Cumulative penalty with smoother growth
    # Use a gentler saturation: x / (1 + 2*x) grows slower than x/(1+x)
    # This avoids over-penalizing edges that have been used a few times
    cum_sat = edge_n_used / (1.0 + 2.0 * edge_n_used)
    
    # Apply squared inverse distance weighting to the cumulative penalty
    # Adjust alpha to account for the change in weight distribution
    # Short edges now have much higher relative weight
    alpha_cum = 0.35
    penalty = alpha_cum * max_d * cum_sat * inv_d_sq_norm
    
    # 3. Current tour penalty
    # Stronger penalty for edges currently in the best local optimum to force 2-opt to drop them
    # Also weighted by squared inverse distance
    alpha_curr = 0.10
    penalty += alpha_curr * max_d * tour_mask * inv_d_sq_norm
    
    # Add penalty to distances
    d_out += penalty
    
    # Ensure symmetry just in case of float errors or asymmetric inputs (though TSP is symmetric)
    # d_out = (d_out + d_out.T) / 2.0 # Optional, input is likely symmetric
    
    # Safety: ensure finite
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
        
    return d_out
