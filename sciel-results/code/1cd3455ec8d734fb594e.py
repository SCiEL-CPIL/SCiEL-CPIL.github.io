"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Computes priority scores for vertices to be used by a greedy MIS decoder.
    The decoder picks vertices in descending order of score.
    
    Heuristic:
    We want to pick vertices that are 'cheap' (low degree) but whose neighbors are 'expensive' (high degree).
    Picking a cheap vertex removes expensive neighbors from the pool, which is beneficial.
    Score = -degree(u) + average_degree_of_neighbors(u)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Convert to numpy if not already, ensure float for calculations
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    # Ensure symmetry? The problem says "bool matrix", usually symmetric for undirected graphs.
    # If not symmetric, we should treat it as directed or symmetrize. 
    # MIS is typically for undirected. Let's assume the input is symmetric or 
    # we should symmetrize to be safe (A + A.T > 0).
    # However, the prompt says "adjacency: bool matrix". 
    # Let's assume it's undirected and symmetric. If not, (A | A.T) is safer.
    # But standard ER graphs are undirected.
    
    deg = adjacency.sum(axis=1).astype(float)
    
    # Calculate average degree of neighbors for each vertex
    # Sum of degrees of neighbors = A @ deg
    neighbor_deg_sum = adjacency @ deg
    
    # Avoid division by zero
    # avg_neighbor_deg = neighbor_deg_sum / (deg + 1e-9)
    # But if deg is 0, avg is 0.
    
    # Heuristic: 
    # Primary: Low degree is good (negative sign)
    # Secondary: High neighbor degree is good (positive sign) because we eliminate "bad" high-degree nodes.
    
    # Let's try a weighted combination.
    # If deg is 0, score should be high (pick isolated nodes first).
    # Formula: score = -deg + alpha * avg_neighbor_deg
    
    # To handle deg=0, we can just use the sum of neighbor degrees directly?
    # No, if deg=0, neighbor_deg_sum=0. Score = 0.
    # If deg=1, neighbor deg=10. Score = -1 + 10 = 9.
    # If deg=10, neighbor deg=1. Score = -10 + 1 = -9.
    # This ordering seems correct.
    
    # Let's use alpha = 1.0 as a starting point.
    # We can also consider the total "cost" of the neighborhood.
    
    scores = -deg + neighbor_deg_sum / (deg + 1e-9)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple LDF if calculation takes too long (unlikely for 200 nodes)
        return -deg

    return scores
# EVOLVE-BLOCK-END
