import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: penalize edges in the current local optimum
    and historically used edges to escape local optima.
    
    Strategy:
    - Strong penalty on edges in the current local optimum tour
    - Smooth tanh-scaled penalty on historically used edges
    - Degree-based topological bias to encourage balanced degree distribution
    - Keep matrix finite and symmetric
    """
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, leave 20s safety margin
    
    n = edge_distance.shape[0]
    
    # Ensure we work with float64 for numerical stability
    dist = edge_distance.astype(np.float64, copy=True)
    usage = edge_n_used.astype(np.float64, copy=True)
    
    # Make usage symmetric (in case it isn't perfectly)
    usage = 0.5 * (usage + usage.T)
    
    # Compute scale factors
    mean_dist = np.mean(dist)
    max_usage = np.max(usage) if np.max(usage) > 0 else 1.0
    
    # 1. Strong penalty on edges in the current local optimum tour
    # The tour is a list of nodes in order; edges are (tour[i], tour[(i+1)%n])
    tour = np.array(local_opt_tour)
    tour_len = len(tour)
    
    # Base penalty for current tour edges: make them expensive to discourage reuse
    tour_penalty_weight = 0.35 * mean_dist
    
    # Apply penalty to each edge in the current tour
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        dist[u, v] += tour_penalty_weight
        dist[v, u] += tour_penalty_weight
    
    # 2. Smooth tanh-scaled usage penalty
    # tanh(usage / scale) gives a smooth, bounded penalty that saturates
    usage_scale = max(1.0, max_usage * 0.5)
    smooth_usage = np.tanh(usage / usage_scale)
    
    # Scale the usage penalty: moderate weight
    usage_penalty_weight = 0.15 * mean_dist
    dist += usage_penalty_weight * smooth_usage
    
    # 3. Degree-based topological bias
    # Penalize edges incident to high-degree nodes in the usage graph
    # This encourages a more balanced tour structure
    degree = np.sum(usage > 0, axis=1)  # count of non-zero usage per node
    max_degree = np.max(degree) if np.max(degree) > 0 else 1.0
    
    # Create a degree bias matrix: higher for edges connecting high-degree nodes
    # Use a smooth version: bias proportional to product of normalized degrees
    norm_degree = degree / max_degree
    degree_bias = np.outer(norm_degree, norm_degree)
    
    # Small weight for degree bias to not dominate
    degree_penalty_weight = 0.05 * mean_dist
    dist += degree_penalty_weight * degree_bias
    
    # Ensure symmetry
    dist = 0.5 * (dist + dist.T)
    
    # Ensure no negative values (distances should be positive)
    dist = np.maximum(dist, 0.0)
    
    # Ensure finite values
    dist = np.where(np.isfinite(dist), dist, mean_dist)
    
    # Check deadline
    if time.time() > deadline:
        # Return a safe default if we're running out of time
        return dist
    
    return dist
