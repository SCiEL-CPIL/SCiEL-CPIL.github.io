"""Guided local search penalty rule for TSP using 2-opt move regret and usage-based penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - 2-opt Move Regret Penalty: For each edge in the current tour, compute the
      minimal length increase required to remove it via a single 2-opt swap.
      Edges with low regret (easy to remove) receive higher penalties, forcing
      the modified 2-opt to preferentially break them.
    - Global Usage-Weighted Penalty: Additionally, penalize ALL edges based on
      their historical usage count in edge_n_used with smooth saturation.
    - Combined effect: creates both local structural disruption and global
      repulsion from historical basins of attraction.
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

    # Calculate the average edge length of the current tour to scale the penalty
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    avg_edge_len = tour_len / n

    # === 2-opt Move Regret Computation ===
    # For each edge (tour[i], tour[i+1]), find the minimum cost increase
    # to remove it via a single 2-opt swap.
    # A 2-opt swap on edges (i,i+1) and (j,j+1) changes tour length by:
    #   delta = d(tour[i],tour[j]) + d(tour[i+1],tour[j+1]) - d(tour[i],tour[i+1]) - d(tour[j],tour[j+1])
    # We want min delta > 0 over all j != i, j != i-1 (mod n)
    
    # Pre-extract tour nodes for speed
    tour_nodes = np.array(local_opt_tour, dtype=np.int64)
    
    # Compute edge lengths for the current tour
    tour_edge_lens = np.zeros(n)
    for i in range(n):
        tour_edge_lens[i] = edge_distance[tour_nodes[i], tour_nodes[(i + 1) % n]]
    
    # For each edge i, compute the minimum regret to remove it
    # regret[i] = min over j of [d(tour[i],tour[j]) + d(tour[i+1],tour[j+1]) - tour_edge_lens[i] - tour_edge_lens[j]]
    # subject to j not adjacent to i
    regret = np.full(n, np.inf)
    
    for i in range(n):
        ni = tour_nodes[i]
        ni1 = tour_nodes[(i + 1) % n]
        ei = tour_edge_lens[i]
        
        # Try all j not adjacent to i
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            nj = tour_nodes[j]
            nj1 = tour_nodes[(j + 1) % n]
            ej = tour_edge_lens[j]
            
            delta = edge_distance[ni, nj] + edge_distance[ni1, nj1] - ei - ej
            if delta > 0:
                if delta < regret[i]:
                    regret[i] = delta
    
    # Handle edges with no valid 2-opt move (shouldn't happen for n>=4)
    # For those, set regret to a large value (hard to remove)
    max_regret = np.max(regret[regret < np.inf]) if np.any(regret < np.inf) else avg_edge_len
    regret[regret == np.inf] = max_regret
    
    # Normalize regret to [0, 1]: low regret = 0, high regret = 1
    min_regret = np.min(regret)
    regret_range = max_regret - min_regret
    if regret_range < 1e-12:
        regret_norm = np.zeros(n)
    else:
        regret_norm = (regret - min_regret) / regret_range
    
    # Penalty for tour edges based on regret:
    # Low regret (easy to remove) -> high penalty
    # High regret (hard to remove) -> low penalty
    # penalty_i = alpha_regret * avg_edge_len * (1 - regret_norm[i])
    alpha_regret = 1.5
    tour_edge_penalty = alpha_regret * avg_edge_len * (1.0 - regret_norm)
    
    # Apply regret-based penalty to the current tour's edges
    for i in range(n):
        u = tour_nodes[i]
        v = tour_nodes[(i + 1) % n]
        D[u, v] += tour_edge_penalty[i]
        D[v, u] += tour_edge_penalty[i]

    # === Global Usage-Based Penalty (smooth saturation) ===
    # Use smooth saturation: count / (1 + count) instead of quadratic
    # This avoids discontinuities and provides meaningful scaling
    usage_penalty = edge_n_used / (1.0 + edge_n_used.astype(np.float64))
    
    # Scale by average edge length
    alpha_usage = 0.8
    penalty_matrix = alpha_usage * avg_edge_len * usage_penalty
    
    D += penalty_matrix

    # Ensure symmetry
    D = (D + D.T) / 2.0
    
    # Ensure finiteness
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
