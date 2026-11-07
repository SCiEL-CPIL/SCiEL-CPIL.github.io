import numpy as np
import time

# Global state for timing and call tracking
_start_time = None
_call_count = 0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search using Additive Quadratic Penalties.
    
    Strategy:
    1. Compute an additive penalty for each edge based on its usage history.
    2. The penalty is proportional to the square of the usage count (edge_n_used^2).
       This creates a super-linear cost for repeatedly used edges, strongly encouraging
       diversification without completely forbidding edges.
    3. Add a fixed, moderate penalty to edges in the current local optimal tour to
       guarantee escape from the current basin.
    4. The penalty magnitude is scaled relative to the mean edge distance to ensure
       the perturbed matrix remains "geometrically plausible" for 2-opt, preventing
       the creation of artificial long shortcuts that 2-opt might incorrectly exploit.
    """
    global _call_count
    global _start_time
    
    _call_count += 1
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return original matrix if we are running out of time
    if time.time() - _start_time > 325:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # 1. Calculate base scale for penalties
    # Use the mean of the upper triangle (excluding diagonal) as a reference distance
    # This ensures penalties are comparable to actual edge lengths
    mask = ~np.eye(n, dtype=bool)
    mean_dist = np.mean(edge_distance[mask])
    if mean_dist == 0 or not np.isfinite(mean_dist):
        mean_dist = 1.0

    # 2. Compute additive penalties
    # Penalty for cumulative usage: alpha * (count^2) * scale
    # We use a quadratic term to make frequent edges increasingly expensive.
    # The scale factor is tuned so that after ~5-10 uses, the penalty is significant 
    # but not dominating the entire distance matrix.
    # Let's target a penalty of ~0.2 * mean_dist after 5 uses.
    # alpha * 25 * scale = 0.2 * mean_dist
    # scale = 0.1 * mean_dist
    # alpha = 0.008
    
    scale = 0.1 * mean_dist
    alpha_quad = 0.008
    
    # Quadratic penalty term
    # edge_n_used is symmetric, so we can square it directly
    # We clip the usage count to prevent extreme values from causing overflow or
    # excessively large penalties that break the metric structure
    clipped_usage = np.clip(edge_n_used, 0, 20)
    quad_penalty = alpha_quad * (clipped_usage ** 2) * scale

    # 3. Immediate penalty for edges in the current local optimum tour
    # This ensures we don't just stick to the current best local solution.
    # A fixed additive penalty of ~0.5 * mean_dist is strong enough to break 
    # the tour but small enough to keep the 2-opt neighborhood valid.
    tour_penalty_val = 0.5 * mean_dist
    
    tour_mask = np.zeros((n, n), dtype=float)
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                tour_mask[u, v] = 1.0
                tour_mask[v, u] = 1.0
    except Exception:
        # If tour is malformed, just use the cumulative penalty
        pass

    immediate_penalty = tour_penalty_val * tour_mask

    # 4. Construct the perturbed distance matrix
    # d_out = d_true + penalty
    d_out = edge_distance + quad_penalty + immediate_penalty

    # 5. Safety checks
    # Ensure symmetry (should be true if inputs are symmetric, but good to enforce)
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure diagonals are 0 (if they were non-zero in input, they should stay 0 for TSP)
    # Actually, TSP distance matrices usually have 0 on diagonal. 
    # If input has non-zero diagonal, we should probably keep it 0 for the perturbed one too,
    # or at least ensure it doesn't create negative loops.
    np.fill_diagonal(d_out, 0.0)

    # Ensure finite values
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()

    # Ensure non-negative distances (though 2-opt works with any real numbers,
    # negative distances can be weird. Since d_true > 0 and penalty >= 0, 
    # d_out should be > 0 unless d_true was negative which is impossible for Euclidean).
    # But let's clip just in case of numerical issues
    d_out = np.maximum(d_out, 0.0)

    return d_out
