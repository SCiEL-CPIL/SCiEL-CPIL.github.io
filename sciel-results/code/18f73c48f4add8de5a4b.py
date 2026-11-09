import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    modified = edge_distance.copy()
    
    max_dist = np.max(edge_distance)
    
    # 1. Penalize edges in the current local optimum strongly
    # This forces the next 2-opt to break the current structure
    current_penalty = max_dist * 5.0
    
    n = len(local_opt_tour)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        modified[u, v] += current_penalty
        modified[v, u] += current_penalty
        
    # 2. Add a smaller penalty for edges used in previous local optima
    # This helps avoid cycling back to previous solutions
    if edge_n_used.max() > 0:
        history_penalty = max_dist * 0.1
        modified += history_penalty * edge_n_used
        
    return modified
