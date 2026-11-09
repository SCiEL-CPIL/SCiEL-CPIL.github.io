"""Guided local search penalty rule for the TSP with targeted basin-escape penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy: Targeted Basin-Escape
    1. Identify edges in the current local_opt_tour.
    2. Apply a strong additive penalty to these current tour edges to break the local optimum.
    3. Apply a smaller additive penalty to edges NOT in the current tour but with high usage counts.
       This prevents the search from falling into other known local optima (basins) that 
       have been visited in previous rounds, without distorting the entire landscape.
    4. The penalty magnitude is scaled by the average edge length to remain geometrically relevant.
    """
    # Check deadline: if we are too close to the limit, return a safe, 
    # minimally perturbed matrix to avoid timeout.
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    D = edge_distance.copy()

    # Calculate a scale factor based on the average edge length of the current tour.
    # This ensures the penalty is proportional to the problem's geometry.
    tour_len = 0.0
    current_tour_edges = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
        # Mark current tour edges
        current_tour_edges[u, v] = True
        current_tour_edges[v, u] = True
    
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # If the tour length is zero or invalid, use a default scale
    if not np.isfinite(avg_edge_len) or avg_edge_len <= 0:
        avg_edge_len = 1.0

    # Define penalty coefficients
    # alpha_current: Strong penalty for edges in the current tour to force 2-opt to change them.
    # alpha_history: Moderate penalty for edges with high usage but not in current tour,
    #                to steer away from other known basins.
    
    # We want the current tour edges to be significantly more expensive than average.
    # A factor of 0.5 * avg_edge_len per usage count is aggressive but effective for escaping.
    alpha_current = 0.5 * avg_edge_len
    
    # For historical edges, we use a smaller factor to avoid over-penalizing the entire space.
    # We only apply this to edges with usage > 0 and not in the current tour.
    alpha_history = 0.1 * avg_edge_len

    # 1. Apply penalty to current tour edges
    # We add penalty proportional to usage count for these specific edges.
    # Since they are in the current tour, their usage count is at least 1 (accumulated in this round).
    # Using the accumulated count makes the penalty stronger if the tour persists.
    D[current_tour_edges] += alpha_current * edge_n_used[current_tour_edges]

    # 2. Apply penalty to historical edges NOT in the current tour
    # Create a mask for edges with usage > 0 AND not in current tour
    historical_mask = (edge_n_used > 0) & (~current_tour_edges)
    
    if np.any(historical_mask):
        # Apply the history penalty
        D[historical_mask] += alpha_history * edge_n_used[historical_mask]

    # Ensure diagonal is 0 (just in case, though TSP doesn't use self-loops)
    np.fill_diagonal(D, 0.0)

    # Ensure symmetry and finiteness
    if not np.all(np.isfinite(D)):
        # Fallback to true distances if something went wrong
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
