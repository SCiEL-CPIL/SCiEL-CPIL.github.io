"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Approach: Local Graph-Theoretic Heuristic
    # Score vertices based on the sum of the degrees of their neighbors.
    # This favors vertices that are peripheral (low degree) but connected to
    # high-degree "core" vertices. Picking such vertices removes the core
    # vertices from consideration, which is beneficial because core vertices
    # have many conflicts.
    
    # Ensure input is a numpy array of floats for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # 1. Calculate the degree of each vertex
    degrees = adj.sum(axis=1)
    
    # 2. Calculate the sum of degrees of the neighbors for each vertex
    # neighbor_deg_sum[i] = sum(degrees[j] for all j in neighbors of i)
    neighbor_deg_sum = adj @ degrees
    
    # 3. Compute the score
    # We want to maximize the neighbor degree sum (removing high-degree neighbors)
    # and minimize the own degree (fewer vertices removed).
    # The difference captures the "efficiency" of picking this vertex.
    scores = neighbor_deg_sum - degrees
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
