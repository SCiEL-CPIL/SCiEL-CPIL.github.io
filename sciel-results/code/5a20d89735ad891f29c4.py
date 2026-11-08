"""Guided local search penalty rule for the TSP with adaptive additive penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - Apply an additive penalty to edges in the current local_opt_tour.
    - The penalty magnitude is adaptive: proportional to the average edge length 
      of the current tour, ensuring the penalty is significant relative to the 
      problem scale.
    - The penalty grows linearly with the number of times an edge has been used 
      in past local optima (edge_n_used).
    - This additive approach is preferred over multiplicative for TSP as it 
      more effectively breaks the local optimum basin without excessively 
      distorting the geometric structure of non-tour edges.
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
    # This ensures the penalty is meaningful regardless of the specific instance scale
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    avg_edge_len = tour_len / n
    
    # Base penalty factor
    # We want the penalty to be substantial. 
    # Using 0.5 * avg_edge_len means a used edge becomes ~1.5x more expensive.
    # If used multiple times, it becomes even more expensive.
    base_penalty = 0.5 * avg_edge_len
    
    # Apply additive penalties to the current tour edges
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        count = edge_n_used[u, v]
        
        # Penalty grows with usage
        # If count is 0 (shouldn't happen for current tour edges as they were just added 
        # in the evaluator's loop, but just in case), we still penalize to break the current loop
        # Actually, the evaluator adds current tour edges to edge_n_used BEFORE calling this?
        # "add the current tour's edges to edge_n_used, call your function"
        # So count >= 1 for all current tour edges.
        
        penalty = base_penalty * (1.0 + 0.5 * count)
        
        D[u, v] += penalty
        D[v, u] += penalty

    # Ensure symmetry and finiteness
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
