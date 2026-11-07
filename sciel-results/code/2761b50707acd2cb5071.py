"""Guided local search penalty rule for TSP with adaptive kick and density-aware crowding."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.monotonic()
_DEADLINE = 330.0  # Kill limit is 330s, use 325s as safety margin


def _elapsed():
    return time.monotonic() - _START_TIME


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Estimate current round number from edge_n_used.
    2. Compute adaptive penalty strength: high early (escape), decays later (converge).
    3. Apply penalty to edges used in the current local optimum tour, weighted by usage count.
    4. Add a small density-aware crowding cost to encourage exploration of sparser regions.
    """
    n = edge_distance.shape[0]
    
    # Deadline guard: if we're close to the limit, return a safe modified matrix quickly
    if _elapsed() > _DEADLINE - 10:
        # Minimal penalty to stay safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
        if max_used > 0:
            return edge_distance * (1.0 + 0.05 * edge_n_used / max_used)
        return edge_distance.copy()
    
    # Estimate round number: the max value in edge_n_used gives a rough indication
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used == 0:
        # First round: apply a moderate uniform penalty to encourage exploration
        # Use a small random perturbation to break symmetry
        rng = np.random.default_rng(hash(tuple(local_opt_tour[:10])) % (2**32))
        noise = rng.random((n, n)) * 0.01
        # Symmetrize
        noise = (noise + noise.T) / 2.0
        np.fill_diagonal(noise, 0.0)
        return edge_distance + noise
    
    # Adaptive kick magnitude:
    # - Early rounds (low max_used): stronger penalty to escape local optima
    # - Later rounds (high max_used): weaker penalty to allow convergence
    # Use a smooth decay: kick = base / (1 + alpha * round_estimate)
    round_estimate = max_used  # Rough proxy for how many rounds have passed
    base_kick = 0.5  # Initial penalty strength
    alpha = 0.15  # Decay rate
    kick = base_kick / (1.0 + alpha * round_estimate)
    
    # Normalize edge_n_used to [0, 1] range for scaling
    if max_used > 0:
        normalized_usage = edge_n_used / max_used
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Compute the penalty matrix: penalize edges that are in the current tour
    # and have been used many times
    penalty_matrix = np.zeros_like(edge_distance)
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        penalty_matrix[i, j] = kick * (1.0 + normalized_usage[i, j])
        penalty_matrix[j, i] = penalty_matrix[i, j]
    
    # Density-aware crowding cost:
    # For each edge (i, j), compute a measure of "crowding" around the midpoint.
    # We approximate this using the distance matrix: count how many points are
    # close to both i and j (i.e., in the intersection of their neighborhoods).
    # This is a proxy for spatial density around the edge's midpoint.
    
    # Use a threshold based on the median distance
    median_dist = np.median(edge_distance[edge_distance > 0]) if n > 1 else 1.0
    threshold = 0.5 * median_dist  # Points within this distance are considered "close"
    
    # For efficiency, compute crowding only for edges in the current tour
    crowding_cost = np.zeros_like(edge_distance)
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        # Count points that are close to both i and j
        close_to_i = (edge_distance[i, :] < threshold)
        close_to_j = (edge_distance[j, :] < threshold)
        crowding = np.sum(close_to_i & close_to_j)
        # Penalty inversely proportional to crowding: sparse regions get less penalty
        # So we want to DISCOURAGE edges in crowded regions, meaning we add MORE penalty there
        # crowding_cost should be higher where crowding is high
        crowding_cost[i, j] = 0.05 * (crowding / n)
        crowding_cost[j, i] = crowding_cost[i, j]
    
    # Combine: base distance + penalty for used tour edges + crowding cost
    modified = edge_distance + penalty_matrix + crowding_cost
    
    # Ensure symmetry
    modified = (modified + modified.T) / 2.0
    np.fill_diagonal(modified, 0.0)
    
    # Ensure finite values
    modified = np.where(np.isfinite(modified), modified, edge_distance)
    
    return modified
# EVOLVE-BLOCK-END
