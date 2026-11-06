"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _vectorized_greedy_completion_cost(current_node, unvisited_nodes, distance_matrix):
    """
    Estimate the cost of completing the tour from current_node through all unvisited_nodes
    using a vectorized nearest-neighbor approach.
    This is O(N^2) but with very low constant factors due to NumPy operations.
    """
    if len(unvisited_nodes) == 0:
        return 0.0
    
    # We simulate the rest of the tour greedily.
    # Instead of a loop with np.delete (which is O(N) per step, making it O(N^2) with high overhead),
    # we can use a mask or just iterate, but since N=50, even a Python loop is fast.
    # However, we can vectorize the "selection" part if we just want the sum of distances.
    # But the path depends on previous choices, so it's inherently sequential.
    
    # Optimization: Since N is small (<=50), a simple loop is fine, but we can optimize the inner distance lookup.
    # Let's stick to a fast loop but ensure we use NumPy for distance access.
    
    remaining = list(unvisited_nodes)
    curr = current_node
    total_cost = 0.0
    
    for _ in range(len(remaining)):
        if not remaining:
            break
        # Find nearest neighbor in remaining
        # distance_matrix[curr, remaining] is a vector operation
        dists = distance_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        total_cost += dists[min_idx]
        next_node = remaining[min_idx]
        remaining.pop(min_idx)
        curr = next_node
        
    return total_cost

def _select_next_node_local_density(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a local density metric:
    1. Cost to enter the node from current.
    2. Cost to exit the node to its nearest unvisited neighbor (proxy for future connectivity).
    3. A penalty based on how "isolated" the node is in the remaining set (optional, maybe just nearest neighbor distance).
    
    Actually, a better local metric for TSP construction is often:
    Score = dist(current, candidate) + alpha * (min dist from candidate to other unvisited)
    This encourages choosing nodes that are close to the current position AND well-connected to the rest,
    avoiding "dead ends" or isolated clusters.
    
    However, the most effective simple heuristic is often just Nearest Neighbor (NN) or Regret-2.
    The prompt asks for a "nearest-neighbor gap" metric estimating incremental cost.
    
    Let's refine: 
    For each candidate c in unvisited:
    1. d_in = dist(current, c)
    2. d_out_min = min(dist(c, u) for u in unvisited if u != c)
    3. If d_out_min is very large, it's a bad choice (isolated).
    4. Score = d_in + beta * d_out_min
    
    This is fully vectorizable.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # 1. Distances from current to all candidates
    d_in = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distances from each candidate to all other unvisited nodes
    # dist_matrix_candidates = distance_matrix[unvisited_nodes, :][:, unvisited_nodes]
    # This creates an NxN matrix where N is len(unvisited_nodes).
    # For N=50, this is 2500 floats, very fast.
    
    # Get the submatrix of distances between unvisited nodes
    dist_submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We want the minimum distance from each candidate to any OTHER unvisited node.
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(dist_submatrix, np.inf)
    
    d_out_min = np.min(dist_submatrix, axis=1)
    
    # Handle case where a node might be the only one (already handled) or if there are issues with inf
    # If a node is isolated (d_out_min is inf), it's a very bad choice unless it's the last one.
    # But in a complete graph, d_out_min should be finite.
    
    # Combine scores
    # We want to minimize the total estimated cost.
    # A common heuristic is d_in + 0.5 * d_out_min
    # Let's try a simple weighted sum.
    weights_in = 1.0
    weights_out = 0.5
    
    scores = d_in + weights_out * d_out_min
    
    # Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    best_candidate = unvisited_nodes[best_idx]
    
    return int(best_candidate)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # Use local density strategy
    next_node = _select_next_node_local_density(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
