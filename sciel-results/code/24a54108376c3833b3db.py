import numpy as np
import time

def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    
    Strategy: Reward vertices that block high-degree neighbors (hubs)
    while penalizing the vertex's own degree (opportunity cost).
    
    score(v) = avg_degree_of_neighbors(v) - degree(v)
    
    This encourages picking low-degree vertices adjacent to high-degree
    vertices, effectively blocking hubs cheaply.
    """
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=np.int64)
    n = adj.shape[0]
    
    # Degrees
    deg = adj.sum(axis=1)
    
    # Sum of neighbor degrees: for each vertex v, sum of deg(u) for u in N(v)
    # This is adj @ deg
    sum_neighbor_deg = adj @ deg
    
    # Average neighbor degree (guard against division by zero)
    safe_deg = np.where(deg > 0, deg, 1)
    avg_neighbor_deg = sum_neighbor_deg / safe_deg
    
    # Score: reward blocking high-degree neighbors, penalize own degree
    score = avg_neighbor_deg - deg.astype(float)
    
    # For isolated vertices (deg=0), they should have high priority
    # (they don't block anything, so always safe to pick)
    # avg_neighbor_deg for isolated = 0, deg = 0, so score = 0
    # That's fine, they'll be picked when their turn comes
    
    return score
