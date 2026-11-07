import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Hard exclusion penalty for Guided Local Search.
    
    Strategy:
    1. Identify edges in the current local optimum tour.
    2. Set the cost of these specific edges to a very large value (1e6).
    
    This approach effectively removes the current tour's edges from consideration
    during the perturbed 2-opt phase. Unlike multiplicative penalties which might
    only slightly alter the ranking of edges, this hard constraint forces the 2-opt
    to find a tour that does not use any of the current tour's edges. This guarantees
    a significant structural change (a "kick" out of the local optimum) rather than
    a minor adjustment, which is often necessary to escape deep local minima in TSP.
    """
    # Safety check: if no usage data or invalid inputs, return original
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    # Create a copy of the distance matrix to modify
    modified_distance = edge_distance.copy()
    
    # Get the number of nodes
    n = len(local_opt_tour)
    if n < 2:
        return modified_distance
    
    # Extract edges from the tour
    # local_opt_tour is a list of node indices, e.g., [0, 1, 2, ..., 99]
    # Note: The problem statement says "closed-tour", and typically 2-opt works on a path
    # where the closing edge is implicit or explicit. 
    # The current program treats it as a list of nodes forming a cycle.
    # Let's assume local_opt_tour is a permutation of 0..N-1.
    # Edges are (tour[i], tour[i+1]) for i in 0..N-2 and (tour[N-1], tour[0]).
    
    # We penalize ALL edges in the current tour.
    # The directive says: "setting the modified distance of all edges in the current 
    # local-optimal tour to np.inf (or a very large finite value)"
    
    # Use a large finite value to avoid numerical issues with infinity in some contexts,
    # though np.inf is usually fine. 1e6 is safe for unit square distances (max ~1.4).
    LARGE_VAL = 1e6
    
    for i in range(n - 1):
        u = local_opt_tour[i]
        v = local_opt_tour[i + 1]
        modified_distance[u, v] = LARGE_VAL
        modified_distance[v, u] = LARGE_VAL
        
    # Add the closing edge
    u = local_opt_tour[-1]
    v = local_opt_tour[0]
    modified_distance[u, v] = LARGE_VAL
    modified_distance[v, u] = LARGE_VAL
    
    return modified_distance
