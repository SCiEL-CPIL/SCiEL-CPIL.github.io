"""Rewritten program: TSP constructive heuristic with Adaptive Centroid and Destination Bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Nearest Neighbor heuristic enhanced with
    adaptive attraction to the centroid of remaining nodes and the destination.
    
    Cost(c) = d(current, c) + lambda * d(c, centroid) + mu * d(c, destination)
    
    This structural change avoids the O(N^3) cost of multi-step lookahead and
    instead uses global geometric properties (centroid) to prevent the tour
    from expanding too far or creating long closing edges.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        # Exact 2-node optimization
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    if _is_deadline_passed():
        # Fallback to pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate centroid of unvisited nodes
    # Note: distance_matrix indices correspond to node IDs.
    # We assume the nodes are indexed 0..N-1.
    # Centroid in terms of node indices is not directly meaningful for Euclidean distance
    # unless we have coordinates. We only have the distance matrix.
    # Therefore, we cannot calculate a geometric centroid.
    
    # Alternative: Use "Median Node" or "Mean Distance to Others".
    # A proxy for "central" is the node that minimizes the sum of distances to all other unvisited nodes.
    # Let's compute the "connectivity cost" for each candidate c:
    # sum_{u in U, u != c} d(c, u)
    
    # Submatrix of distances among unvisited nodes
    # This is O(N^2) to compute, which is acceptable.
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Sum of distances from each candidate to all other unvisited nodes
    # Exclude self (diagonal is 0 anyway)
    connectivity_costs = np.sum(submatrix, axis=1)
    
    # Distances to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive weights
    # N is total nodes? We don't know N directly, but we can infer or use n_unvisited.
    # Let's assume total N is roughly 50 based on task description.
    # Or we can just scale based on n_unvisited.
    # As n_unvisited decreases, we want to be more careful.
    
    # Let's define progress = 1 - n_unvisited / 50.0 (assuming 50 cities)
    # If n_unvisited is 50, progress = 0.
    # If n_unvisited is 1, progress = 0.98.
    
    # We want lambda (connectivity) and mu (destination) to increase.
    # Start with small values to allow greedy exploration.
    
    # Normalize connectivity costs to be comparable to edge lengths?
    # The edge lengths are d(current, c).
    # Connectivity costs are sums of ~n_unvisited edges. They will be larger.
    # Let's divide by n_unvisited to get average distance to others.
    avg_connectivity = connectivity_costs / max(1, n_unvisited - 1)
    
    # Scale factors
    # We want the penalty to be significant but not overwhelming.
    # Let's try lambda = 0.2 + 0.3 * progress
    # mu = 0.2 + 0.3 * progress
    
    # Estimate total nodes N. If not provided, assume 50.
    # The function signature doesn't give N. We can infer N from distance_matrix.shape[0].
    N = distance_matrix.shape[0]
    progress = 1.0 - (n_unvisited / N)
    
    lambda_c = 0.1 + 0.4 * progress
    mu_d = 0.1 + 0.4 * progress
    
    # Total score for each candidate
    # Lower is better
    scores = dists_from_current + lambda_c * avg_connectivity + mu_d * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
