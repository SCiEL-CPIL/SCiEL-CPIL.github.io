import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    Returns a perturbed distance matrix that encourages 2-opt to escape local optima.
    
    Strategy:
    - Penalize edges in the current tour strongly (to force 2-opt to break them)
    - Penalize frequently-used edges with a nonlinear (power 1.5) scaling
    - Use a saturating exponential for additional nonlinearity
    - Combine penalties with careful scaling
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Initialize with true distances
    d_out = edge_distance.astype(np.float64).copy()
    
    # Find max distance to scale penalties
    max_d = np.max(d_out)
    if max_d == 0:
        return d_out
    
    # 1. Build tour mask for edges in current local optimum
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
    
    # 2. Compute cumulative penalty using saturating function
    # edge_n_used[i,j] is the count of times edge (i,j) has been used
    # Use a smooth saturating function: x / (1 + x) maps [0, inf) to [0, 1)
    # This ensures penalties grow but don't explode
    
    # Scale factor: we want the penalty to be significant but not overwhelming
    # For an edge used ~5-10 times, saturation is ~0.8-0.9
    # Target max historical penalty ~ 0.3 * max_d
    cum_sat = edge_n_used / (1.0 + edge_n_used)
    alpha_cum = 0.25
    penalty_cum = alpha_cum * max_d * cum_sat
    
    # 3. Current tour penalty
    # Stronger penalty for edges currently in the best local optimum
    # This forces 2-opt to drop these edges more aggressively
    alpha_curr = 0.15
    penalty_curr = alpha_curr * max_d * tour_mask
    
    # 4. Repulsive force model: penalty inversely proportional to edge distance
    # Short edges (high value, frequently used in TSP) get larger penalties
    # This creates a non-linear, geometry-aware weighting that forces exploration
    # of alternative short connections
    
    # Compute inverse distance weighting: 1/d is large for short edges
    # Avoid division by zero for diagonal
    d_safe = np.where(edge_distance > 1e-10, edge_distance, 1e10)
    inv_dist = 1.0 / d_safe
    # Normalize inv_dist so max is 1.0
    max_inv = np.max(inv_dist)
    if max_inv > 0:
        inv_dist_norm = inv_dist / max_inv
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # Apply repulsive penalty: scaled by usage count and inverse distance
    # Short, frequently-used edges get the largest penalty
    alpha_repulsive = 0.15
    penalty_repulsive = alpha_repulsive * max_d * inv_dist_norm * cum_sat
    
    # Combine all penalty schemes
    penalty = np.maximum(penalty_cum, penalty_repulsive) + penalty_curr
    
    # 5. Add penalty to distances
    d_out += penalty
    
    # 6. Optional: slight random perturbation to break ties
    # This helps escape symmetric local optima
    # Keep it small to not distort the problem too much
    if n <= 200:  # Only for small instances to keep runtime low
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        noise_scale = 0.002 * max_d
        noise = rng.uniform(-noise_scale, noise_scale, size=(n, n))
        # Ensure symmetry
        noise = (noise + noise.T) / 2.0
        np.fill_diagonal(noise, 0.0)
        d_out += noise
    
    # 7. Ensure symmetry
    d_out = (d_out + d_out.T) / 2.0
    
    # 8. Ensure non-negative and finite
    d_out = np.maximum(d_out, 0.0)
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
    
    # Check deadline
    if time.time() > deadline:
        return edge_distance.copy()
    
    return d_out
