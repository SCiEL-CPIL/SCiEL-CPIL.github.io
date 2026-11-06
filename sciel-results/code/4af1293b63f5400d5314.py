"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors for each vertex
    # neighbor_deg_sum[v] = sum of deg[u] for u in N(v)
    neighbor_deg_sum = A @ deg  # matrix-vector product: (A @ deg)[v] = sum_u A[v,u] * deg[u]
    
    # Mean degree of neighbors (avoid division by zero)
    safe_deg = np.maximum(deg, 1.0)
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Score: low degree is good (negative), high mean neighbor degree is good (positive)
    # This rewards picking vertices that are low-degree but adjacent to high-degree vertices
    # because blocking high-degree neighbors is "cheap" in terms of lost potential independent set size
    score = -deg + 0.5 * mean_neighbor_deg
    
    # Check deadline
    if time.time() > deadline:
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
