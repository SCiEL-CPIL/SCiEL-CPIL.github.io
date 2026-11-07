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
    
    # 2. Compute penalty based on edge usage count
    # Use power 1.5 scaling for more aggressive penalty on frequently used edges
    # This creates a steeper penalty curve than linear scaling
    
    # Method A: Power scaling with saturation
    # f(x) = (x / (1 + x)) ** 0.75 - gives diminishing returns but still grows
    # Method B: Exponential saturation 1 - exp(-x/k)
    # Method C: Pure power x^0.5 normalized
    
    # Combine: use a blend that emphasizes both recent and historical usage
    edge_n_used_float = edge_n_used.astype(np.float64)
    
    # Power 1.5 on the count, then normalize by max count to keep bounded
    max_count = np.max(edge_n_used_float)
    if max_count > 0:
        # Normalize counts to [0, 1]
        normalized_counts = edge_n_used_float / (1.0 + max_count)
        # Apply power 1.5 for super-linear penalty growth
        penalty_power = normalized_counts ** 1.5
    else:
        penalty_power = np.zeros_like(edge_n_used_float)
    
    # Exponential saturation component
    k = 4.0
    penalty_exp = 1.0 - np.exp(-edge_n_used_float / k)
    
    # Combine both penalty schemes: take a weighted average
    # The power component emphasizes high-count edges, the exp component gives smooth growth
    alpha_power = 0.6
    alpha_exp = 0.4
    combined_usage_penalty = alpha_power * penalty_power + alpha_exp * penalty_exp
    
    # Scale the usage penalty
    # Target: an edge used ~10 times should get a penalty of ~0.3 * max_d
    # At count=10, normalized=10/11≈0.91, power=0.91^1.5≈0.87, exp=1-exp(-2.5)≈0.92
    # combined ≈ 0.6*0.87 + 0.4*0.92 ≈ 0.89
    # So scale factor should be ~0.35 to get ~0.31 * max_d
    usage_scale = 0.35
    penalty_usage = usage_scale * max_d * combined_usage_penalty
    
    # 3. Current tour penalty - strong penalty to force breaking current tour edges
    # This is the key GLS mechanism: penalize the current solution's edges
    alpha_curr = 0.20
    penalty_curr = alpha_curr * max_d * tour_mask
    
    # 4. Additional: bonus (negative penalty) for short edges not in tour
    # This encourages 2-opt to consider shorter alternative edges
    # Only apply to edges that are NOT in the tour and have low usage
    short_edge_threshold = 0.3 * max_d
    short_mask = (d_out < short_edge_threshold) & (tour_mask == 0) & (edge_n_used_float < 3)
    bonus_scale = 0.05
    penalty_bonus = -bonus_scale * max_d * short_mask
    
    # 5. Add all penalties to distances
    d_out += penalty_usage + penalty_curr + penalty_bonus
    
    # 6. Ensure symmetry
    d_out = (d_out + d_out.T) / 2.0
    
    # 7. Ensure non-negative and finite
    d_out = np.maximum(d_out, 0.0)
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
    
    # Check deadline
    if time.time() > deadline:
        return edge_distance.copy()
    
    return d_out
