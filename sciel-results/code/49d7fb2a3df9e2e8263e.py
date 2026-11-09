"""
Route-aware heuristic for ACO CVRP: build edge-attractiveness from
greedy route simulation that captures demand-feasible route structure.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_route_cost(distance_matrix, demands, capacity, alpha=1.0):
    """
    Run a greedy nearest-neighbor with demand feasibility to compute
    total route cost and collect edge frequencies.
    """
    n = distance_matrix.shape[0]
    unvisited = set(range(1, n))
    total_cost = 0.0
    edge_counts = np.zeros((n, n), dtype=np.float64)
    
    current = 0
    remaining_cap = float(capacity)
    
    while unvisited:
        # Find best next customer that fits
        best_j = -1
        best_score = -1e18
        for j in unvisited:
            if demands[j] <= remaining_cap:
                # Score: inverse distance weighted by demand efficiency
                d = distance_matrix[current, j]
                score = (1.0 / (d + 1e-9)) / (1.0 + alpha * demands[j] / capacity)
                if score > best_score:
                    best_score = score
                    best_j = j
        
        if best_j == -1:
            # No customer fits, return to depot
            total_cost += distance_matrix[current, 0]
            edge_counts[current, 0] += 1.0
            current = 0
            remaining_cap = float(capacity)
        else:
            # Move to best_j
            total_cost += distance_matrix[current, best_j]
            edge_counts[current, best_j] += 1.0
            remaining_cap -= demands[best_j]
            unvisited.discard(best_j)
            current = best_j
    
    # Return to depot
    if current != 0:
        total_cost += distance_matrix[current, 0]
        edge_counts[current, 0] += 1.0
    
    return total_cost, edge_counts


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Route-aware approach:
      - Run multiple greedy simulations with different demand-weighting alphas.
      - Accumulate edge frequencies across all simulations.
      - Combine with inverse distance to get final heuristic.
      - Edges that appear frequently in good greedy routes get higher attractiveness.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s kill

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]

    dist_safe = np.maximum(distance_matrix, 1e-9)
    base = 1.0 / dist_safe

    # Run greedy simulations with different alpha values
    # alpha controls how strongly demand affects the greedy choice
    alphas = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    
    total_edge_counts = np.zeros((n, n), dtype=np.float64)
    total_cost = 0.0
    num_sims = 0
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        cost, edge_counts = _greedy_route_cost(distance_matrix, demands, capacity, alpha)
        total_edge_counts += edge_counts
        total_cost += cost
        num_sims += 1
    
    if num_sims == 0:
        # Fallback if all simulations timed out
        heuristic = base.copy()
    else:
        # Normalize edge counts
        avg_edge_counts = total_edge_counts / num_sims
        
        # Heuristic: combine inverse distance with edge frequency
        # Edges that are both short AND frequently used in good routes get boosted
        # Use a power law to emphasize frequent edges
        edge_freq_factor = 1.0 + 0.5 * avg_edge_counts
        
        heuristic = base * edge_freq_factor

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
