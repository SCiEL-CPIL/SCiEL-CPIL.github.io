"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    Higher score = picked earlier.
    
    Strategy: Use a combination of:
    1. Core number (from k-core decomposition) - lower core = periphery = pick earlier
    2. Degree - lower degree = blocks fewer = pick earlier  
    3. Neighborhood quality - if neighbors have high degree, picking this vertex
       is 'cheaper' since those neighbors would be less likely to be picked anyway
    """
    n = len(adjacency)
    deg = adjacency.sum(axis=1).astype(float)
    
    # Compute mean degree of neighbors for each vertex
    # For vertex v, mean_neighbor_deg = sum(deg[u] for u in N(v)) / deg[v]
    # Use matrix multiplication: (adjacency @ deg) / deg
    neighbor_deg_sum = adjacency @ deg
    # Avoid division by zero
    safe_deg = np.where(deg > 0, deg, 1.0)
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    # For isolated vertices, set to 0
    mean_neighbor_deg = np.where(deg > 0, mean_neighbor_deg, 0.0)
    
    # Score: we want to pick vertices that are:
    # - Low degree (blocks fewer)
    # - Whose neighbors have high degree (picking v is "cheaper")
    # 
    # score = -degree + beta * mean_neighbor_degree
    # Normalize to make it robust
    beta = 0.5
    score = -deg + beta * mean_neighbor_deg
    
    return score
# EVOLVE-BLOCK-END
