import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive multiplicative penalty for Guided Local Search.
    Uses a dynamic alpha that scales with the maximum usage count to
    increase pressure on frequently used edges as the search progresses.
    """
    # Safety check for empty or invalid inputs
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    max_used = np.max(edge_n_used)
    
    if max_used == 0:
        return edge_distance.copy()
    
    # Adaptive alpha: starts small, grows as edges are reused
    # Using a square root scaling to avoid explosive growth but ensure
    # that later rounds have significantly higher penalties than early ones.
    # Base alpha is tuned to be effective for TSP distances in [0, ~1.5]
    alpha = 0.1 * np.sqrt(max_used)
    
    # Multiplicative penalty: 
    # This preserves the relative geometry of the distance matrix better than additive,
    # which can distort the Euclidean metric structure more severely.
    # Edges with high usage get a larger relative increase, making them less attractive
    # to the 2-opt heuristic.
    
    # Ensure the penalty is applied only where usage > 0 to avoid unnecessary computation
    # and to strictly penalize only the "stuck" edges.
    penalty_matrix = np.zeros_like(edge_distance)
    mask = edge_n_used > 0
    penalty_matrix[mask] = alpha * edge_n_used[mask]
    
    # Multiplicative update
    # 1 + penalty ensures distances remain positive.
    # The penalty is proportional to usage, so frequently used edges get a larger multiplier.
    return edge_distance * (1.0 + penalty_matrix)
