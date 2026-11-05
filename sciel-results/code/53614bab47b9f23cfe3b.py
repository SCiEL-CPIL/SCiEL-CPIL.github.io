import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin out of 120s
    
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.float64)
        
    n = adjacency.shape[0]
    
    # Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # Handle case where graph is empty or isolated nodes
    if n == 0:
        return np.array([])
    
    # 1. Feature: Degree (normalized)
    max_deg = np.max(degrees) if np.max(degrees) > 0 else 1.0
    deg_norm = degrees / max_deg
    
    # 2. Feature: Eigenvector Centrality (Global Spectral Property)
    # Power Iteration for the principal eigenvector of A
    # This captures "influence" or "centrality" in the graph.
    # Low centrality nodes are often better for MIS as they are less "central" to conflicts.
    
    # Initialize with random or ones
    x = np.ones(n, dtype=np.float64)
    x = x / np.linalg.norm(x)
    
    # Power Iteration steps
    # Use a few iterations to converge. 50 is usually enough for N=200.
    for _ in range(50):
        x_new = adjacency @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x = x_new / norm
        
    # Eigenvector centrality scores
    eig_centrality = x
    max_eig = np.max(eig_centrality) if np.max(eig_centrality) > 0 else 1.0
    eig_norm = eig_centrality / max_eig
    
    # 3. Feature: Local Neighbor Strength (Average Neighbor Degree)
    # Same as before, but normalized
    sum_neighbor_degrees = adjacency @ degrees
    avg_neighbor_degrees = np.zeros(n, dtype=float)
    mask = degrees > 0
    avg_neighbor_degrees[mask] = sum_neighbor_degrees[mask] / degrees[mask]
    
    max_avg_deg = np.max(avg_neighbor_degrees) if np.max(avg_neighbor_degrees) > 0 else 1.0
    avg_deg_norm = avg_neighbor_degrees / max_avg_deg
    
    # Combine features
    # We want to MINIMIZE the score for the greedy picker? 
    # No, the greedy picker picks DESCENDING score.
    # So we want GOOD vertices to have HIGH scores.
    # Good vertices have:
    # - Low degree (so -deg)
    # - Low eigenvector centrality (so -eig)
    # - Low average neighbor degree (so -avg_deg)
    
    # Weights:
    # Degree is the strongest signal.
    # Eigenvector adds global context.
    # Avg neighbor degree adds local context.
    
    alpha = 1.0
    beta = 0.5
    gamma = 0.5
    
    scores = -(alpha * deg_norm + beta * eig_norm + gamma * avg_deg_norm)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple degree if calculation took too long
        return -degrees.astype(float)
        
    return scores
