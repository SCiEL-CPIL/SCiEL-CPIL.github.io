"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Computes a priority score for each vertex to guide a greedy
    Maximum Independent Set selection.
    
    Heuristic:
    1. Prefer vertices with low degree (low cost to select).
    2. Among those, prefer vertices whose neighbors have high degree 
       (high benefit from blocking "hubs").
    
    Score = -degree(v) + mean_degree(neighbors(v))
    """
    # Ensure adjacency is a numpy array for efficient matrix multiplication
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    deg = adjacency.sum(axis=1)
    
    # Calculate sum of degrees of neighbors for each vertex
    # This is equivalent to A * deg
    sum_neigh_deg = adjacency @ deg
    
    # Calculate mean degree of neighbors
    # Avoid division by zero for isolated vertices
    mean_neigh_deg = np.divide(
        sum_neigh_deg, 
        deg, 
        out=np.zeros_like(deg, dtype=float), 
        where=deg > 0
    )
    
    # Final score
    # We want to maximize this score.
    # Low degree is good (negative penalty).
    # High neighbor degree is good (positive reward).
    score = -deg + mean_neigh_deg
    
    return score
# EVOLVE-BLOCK-END
