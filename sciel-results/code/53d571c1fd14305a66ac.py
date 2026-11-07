import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Rank-based bounded additive penalty for Guided Local Search.
    The penalty for each edge is proportional to its usage count, 
    normalized so that the most frequently used edge receives a 
    fixed maximum penalty. This prevents penalty explosion in later 
    rounds while maintaining strong pressure on the most "stuck" edges.
    """
    # Safety check for empty or invalid inputs
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    # Find the maximum usage count
    max_used = np.max(edge_n_used)
    
    if max_used == 0:
        return edge_distance.copy()
    
    # Define a fixed maximum penalty value.
    # 0.5 is a standard heuristic for TSP in the unit square, 
    # large enough to break local optima but small enough to preserve geometry.
    max_penalty = 0.5
    
    # Calculate normalized penalty matrix
    # Only edges with usage > 0 will have non-zero penalty
    # This scales the penalty linearly with usage, capped at max_penalty for the most used edge
    penalty_matrix = np.zeros_like(edge_distance)
    mask = edge_n_used > 0
    penalty_matrix[mask] = max_penalty * (edge_n_used[mask] / max_used)
    
    # Additive update
    # Adding the penalty increases the cost of frequently used edges,
    # encouraging the 2-opt heuristic to find alternative routes.
    return edge_distance + penalty_matrix
