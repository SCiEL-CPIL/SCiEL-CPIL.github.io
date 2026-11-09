import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    modified = edge_distance.copy()
    
    # 1. Edge-Reversal Heuristic: Force removal of current tour edges
    # By setting the cost of current tour edges to a very high value,
    # we force the subsequent 2-opt to break the current local optimum
    # and explore alternative connections.
    n = len(local_opt_tour)
    high_cost = 1e9
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        modified[u, v] = high_cost
        modified[v, u] = high_cost
        
    # 2. Penalize edges that have been used in previous local optima
    # This helps avoid cycling back to previously visited solutions.
    if edge_n_used.max() > 0:
        max_dist = np.max(edge_distance)
        # Use a penalty factor that scales with the number of times an edge was used.
        # A factor of 0.5 * max_dist per use is significant enough to discourage
        # reuse but not so large that it completely distorts the geometry for
        # edges with low usage counts.
        history_penalty_factor = max_dist * 0.5
        modified += history_penalty_factor * edge_n_used
        
    return modified
