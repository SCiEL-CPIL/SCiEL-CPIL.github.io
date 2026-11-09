import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    modified = edge_distance.copy()
    
    n = len(local_opt_tour)
    
    # Set the cost of all edges in the current local optimal tour to a very large value.
    # This acts as a "hard" penalty, forcing the subsequent 2-opt to break the current structure
    # and explore alternative connections, effectively implementing the edge-reversal heuristic.
    # We use a large finite number instead of infinity to satisfy the "finite matrix" constraint.
    penalty = 1e10
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        modified[u, v] = penalty
        modified[v, u] = penalty
        
    return modified
