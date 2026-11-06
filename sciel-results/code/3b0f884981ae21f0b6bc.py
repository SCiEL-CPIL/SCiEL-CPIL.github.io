import numpy as np

def priority(adjacency):
    # Convert to float for matrix multiplication
    adj_float = adjacency.astype(float)
    
    # Degree of each vertex
    deg = adj_float.sum(axis=1)
    
    # Sum of degrees of neighbors
    # S_v = sum_{u in N(v)} deg[u]
    neighbor_deg_sum = adj_float @ deg
    
    # Average degree of neighbors
    # Handle d(v) = 0 case to avoid division by zero
    # If d(v) = 0, avg is 0.
    # np.where or divide with where
    avg_neighbor_deg = np.zeros_like(deg)
    non_zero = deg > 0
    avg_neighbor_deg[non_zero] = neighbor_deg_sum[non_zero] / deg[non_zero]
    
    # Score: -degree + avg_neighbor_degree
    # This prioritizes low degree vertices, with a boost for those connected to high-degree vertices.
    scores = -deg + avg_neighbor_deg
    
    return scores
