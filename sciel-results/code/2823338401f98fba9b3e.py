"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix


def priority(adjacency):
    """
    Compute vertex priorities for the greedy independent set decoder.
    
    Uses a multi-factor scoring approach:
    1. Base score: negative degree (low degree first)
    2. Penalty for neighbors with high degree (avoid picking vertices adjacent to hubs)
    3. Reward for being in low-density regions
    4. Small random perturbation for tie-breaking
    """
    start_time = time.time()
    deadline = 140.0
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Degree of each vertex
    degrees = adj.sum(axis=1).astype(float)
    
    # Mean degree of neighbors (for each vertex v, average degree of its neighbors)
    # This captures whether v is connected to high-degree vertices
    deg2 = degrees ** 2
    # Sum of neighbor degrees for each vertex
    neighbor_deg_sum = adj @ degrees
    # Mean neighbor degree (0 if no neighbors)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degrees > 0, neighbor_deg_sum / degrees, 0)
    
    # Score: we want to pick vertices with low degree and low mean neighbor degree
    # Higher score = picked first
    # So: negative degree, negative mean neighbor degree
    base_score = -degrees - 0.5 * mean_neighbor_deg
    
    # Add small random perturbation for deterministic tie-breaking
    rng = np.random.default_rng(42)
    scores = base_score + rng.random(n) * 1e-8
    
    return scores
# EVOLVE-BLOCK-END
