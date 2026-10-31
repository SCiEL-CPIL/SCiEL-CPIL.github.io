"""Traveling salesman constructive heuristic: Dynamic Lookahead 2-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching and timing
_start_time = None
_cache = {}

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    if n > 20:
        # Use a few strategic points to hash
        sample = distance_matrix[0, :5].tobytes() + distance_matrix[10, :5].tobytes() + distance_matrix[25, :5].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _fallback_nearest(current_node, unvisited_nodes, distance_matrix):
    """Simple nearest neighbor fallback."""
    if len(unvisited_nodes) == 0:
        return int(current_node)
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    dists = distance_matrix[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])

def _select_next_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix, tour_so_far):
    """
    Select the next node by minimizing the incremental cost of adding it to the tour,
    considering a 2-step lookahead to correct for immediate local suboptimality.
    
    Score for candidate k:
    1. Immediate cost: dist(current, k)
    2. Future cost: min_{j in unvisited \ {k}} dist(k, j) + dist(j, destination)
       (This approximates the cost of the next two edges, helping to avoid dead ends)
    
    We combine these to pick the best candidate.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited <= 2:
        # If very few left, just pick the one that minimizes total return cost
        if n_unvisited == 1:
            return int(unvisited_nodes[0])
        # n_unvisited == 2: pick the one that leads to lower total cost back to destination
        # Cost 1: current -> u0 -> u1 -> dest
        # Cost 2: current -> u1 -> u0 -> dest
        u0, u1 = unvisited_nodes[0], unvisited_nodes[1]
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        if cost0 <= cost1:
            return int(u0)
        else:
            return int(u1)
    
    # Pre-compute distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Pre-compute distances from destination to all unvisited (for lookahead)
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # We want to find the best next node k.
    # For each k, we estimate the cost of the path segment: current -> k -> best_next -> destination
    # best_next is the unvisited node (excluding k) that minimizes dist(k, best_next) + dist(best_next, destination)
    
    # This is O(N^2) in the number of unvisited nodes if done naively.
    # For N=50, this is 2500 operations per step, which is very fast.
    
    best_score = np.inf
    best_node = -1
    
    # Vectorized approach for speed
    # dist_from_current is shape (N,)
    # dist_to_dest is shape (N,)
    
    # For each candidate i in unvisited_nodes:
    # immediate = dist_from_current[i]
    # future = min_{j != i} (distance_matrix[unvisited_nodes[i], unvisited_nodes[j]] + dist_to_dest[j])
    
    # Let's compute a matrix of distances between all unvisited nodes
    # This is expensive to build every step (N^2), but N decreases.
    # However, we can optimize by only looking at a subset if N is large, or just do it directly.
    # Given N=50, building a 50x50 submatrix is trivial.
    
    # Extract submatrix of distances among unvisited nodes
    # Note: unvisited_nodes is an array of indices
    # dist_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # To avoid O(N^2) memory allocation overhead in Python for every step, 
    # and since N is small (<=50), we can just loop or use numpy efficiently.
    
    # Let's try a semi-vectorized approach.
    # For each candidate i, we need min over j of (dist[i,j] + dist_to_dest[j])
    
    # Create a matrix where M[i,j] = dist(unvisited[i], unvisited[j]) + dist_to_dest[j]
    # Then for each i, we want min(M[i, :]) excluding j=i
    
    # dist_sub is (N, N)
    dist_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Add dist_to_dest to each row? No, dist_to_dest[j] depends on the target j.
    # So M[i,j] = dist_sub[i,j] + dist_to_dest[j]
    # We can add dist_to_dest (shape N) to the columns of dist_sub.
    # dist_sub + dist_to_dest[None, :] broadcasts correctly.
    
    M = dist_sub + dist_to_dest[None, :]
    
    # Mask the diagonal (j=i) to infinity so we don't pick the same node
    np.fill_diagonal(M, np.inf)
    
    # For each i, find the min value in M[i, :]
    future_costs = np.min(M, axis=1)
    
    # Total score for candidate i:
    # score[i] = dist_from_current[i] + future_costs[i]
    
    scores = dist_from_current + future_costs
    
    # Find the index of the minimum score
    best_idx = np.argmin(scores)
    best_node = unvisited_nodes[best_idx]
    
    return int(best_node)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using a dynamic lookahead heuristic.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() - _start_time > 145.0:
        # Fallback: nearest neighbor
        return _fallback_nearest(current_node, unvisited_nodes, distance_matrix)
    
    # Use lookahead heuristic
    # tour_so_far is not strictly needed for the calculation, but kept for signature compatibility if needed
    return _select_next_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix, None)
# EVOLVE-BLOCK-END
