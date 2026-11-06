"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    
    Primary criterion: low degree (fewer conflicts).
    Secondary criterion: high mean degree of neighbors (excluding high-degree
    neighbors is less costly since they'd be hard to include anyway).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    degrees = adjacency.sum(axis=1).astype(float)
    # Sum of degrees of neighbors for each vertex
    neighbor_degree_sums = degrees @ adjacency
    # Mean degree of neighbors (avoid division by zero)
    mean_neighbor_degrees = neighbor_degree_sums / (degrees + 1)
    
    # Primary: prefer low degree. Secondary: prefer high mean neighbor degree.
    score = -degrees * 1000.0 + mean_neighbor_degrees
    
    # Time guard
    if time.time() - start_time > deadline:
        # Fallback to simple degree-based score
        return -degrees
    
    return score
# EVOLVE-BLOCK-END
