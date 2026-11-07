"""Guided local search penalty rule for TSP with adaptive multiplicative penalty."""
import numpy as np
import time

# Track wall-clock time for safety
_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP with adaptive multiplicative penalty.
    
    Strategy:
    1. Infer the current search round from the maximum edge usage count.
    2. Normalize edge usage counts using soft saturation to avoid under-penalizing heavily used edges.
    3. Apply a power-law to emphasize frequently used edges.
    4. Calculate an adaptive base penalty coefficient that increases with the round number.
    5. Apply an additional specific penalty to edges present in the current local optimal tour.
    6. Multiply the true distances by the calculated penalty multipliers.
    7. Ensure the resulting matrix is symmetric, finite, and has a zero diagonal.
    """
    # Check time limit; if exceeded, return original distances to avoid timeout
    elapsed = time.time() - _GLOBAL_START_TIME
    if elapsed > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Get edges of the current local optimal tour
    tour = local_opt_tour
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Infer round number: max edge usage indicates how many rounds have passed
    max_used = edge_n_used.max()
    if max_used > 0:
        round_num = int(max_used)
    else:
        round_num = 0
        
    # Normalize edge usage using soft saturation: u / (1 + u)
    # This avoids under-penalizing heavily used edges in later rounds
    # while keeping the penalty bounded
    usage_float = edge_n_used.astype(np.float64)
    norm_used = usage_float / (1.0 + usage_float)
    
    # Power law on normalized usage to emphasize heavy hitters
    # Gamma > 1 makes the penalty super-linear, strongly penalizing very common edges
    gamma = 1.8
    p_norm = np.power(norm_used, gamma)
    
    # Adaptive base alpha: increases penalty strength with rounds to escape deeper local optima
    # Start moderate, ramp up linearly, cap to avoid destroying all structure
    base_alpha = 0.25 + 0.04 * round_num
    base_alpha = min(base_alpha, 1.3)
    
    # Extra penalty for edges in the current tour
    # This is the core GLS mechanism: penalize the specific solution found
    tour_alpha = 0.12 + 0.025 * round_num
    tour_alpha = min(tour_alpha, 0.55)
    
    # Create a mask for edges in the current tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_mask[u, v] = 1.0
        tour_mask[v, u] = 1.0
        
    # Calculate the total penalty multiplier for each edge
    # Base penalty from usage history + extra penalty if in current tour
    penalty_multiplier = 1.0 + base_alpha * p_norm
    penalty_multiplier += tour_alpha * tour_mask
    
    # Clip to reasonable range to prevent extreme distortions
    penalty_multiplier = np.clip(penalty_multiplier, 1.0, 6.0)
    
    # Apply multiplicative penalty
    new_distances = edge_distance * penalty_multiplier
    
    # Ensure symmetry
    new_distances = (new_distances + new_distances.T) / 2.0
    
    # Ensure diagonal is 0 (distance from a node to itself)
    np.fill_diagonal(new_distances, 0.0)
    
    # Ensure finiteness
    if not np.all(np.isfinite(new_distances)):
        new_distances = np.nan_to_num(new_distances, nan=1.0, posinf=1.0, neginf=1.0)
        
    return new_distances
