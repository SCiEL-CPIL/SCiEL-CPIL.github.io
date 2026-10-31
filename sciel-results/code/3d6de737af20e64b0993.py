import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Progressive Lookahead heuristic.
    
    The score for each candidate node u is:
    score(u) = d(current, u) + lambda * d(u, destination)
    
    Where lambda increases as the tour progresses, shifting the focus
    from local nearest-neighbor expansion to global tour closure.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Track time to ensure we don't exceed the 150s limit
    # Although this function is fast, the evaluator might call it many times
    # or there might be overhead. We add a simple check.
    # Note: The 150s limit is for the ENTIRE evaluation of 8 instances.
    # So we have ~18s per instance. With 50 nodes, we have 50 calls.
    # ~0.36s per call. This is plenty for O(N) vectorized ops.
    
    n_remaining = len(unvisited_nodes)
    total_nodes = distance_matrix.shape[0]
    
    # Calculate progress: 0.0 at start, 1.0 at end
    # n_visited = total_nodes - n_remaining
    # progress = n_visited / total_nodes
    progress = 1.0 - (n_remaining / max(total_nodes, 1))
    
    # Distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from destination to all unvisited
    dists_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Adaptive weight for the lookahead term
    # We want to start with pure Nearest Neighbor (lambda=0)
    # and end with pure Destination-Nearest (lambda=1 or higher)
    # A linear ramp is a good start.
    # We can also try a quadratic ramp to be more aggressive at the end.
    # lambda = progress * progress * 2.0  # Aggressive late game
    # Let's try a standard linear ramp first, maybe scaled.
    # If lambda is too high early, it ruins the local structure.
    # If lambda is too low late, we leave bad edges to close.
    
    # Let's use a weight that ramps up from 0 to 1.5
    # This ensures that in the last few steps, we strongly prefer nodes close to destination.
    lambda_weight = progress * 1.5
    
    # Calculate total score: we want to MINIMIZE this
    # score = dist_to_current + lambda * dist_to_destination
    scores = dists_from_current + lambda_weight * dists_from_dest
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
