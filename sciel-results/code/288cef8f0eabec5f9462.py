import numpy as np
import time

def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    
    Strategy:
    - Prioritize vertices with low degree (they block fewer neighbors).
    - Prioritize vertices with high mean neighbor degree (picking them 'removes' high-degree hubs, 
      which would otherwise have low priority but might interfere, or more accurately, 
      it aligns with the 'low degree, high neighbor degree' pattern of leaves attached to hubs, 
      which are optimal to pick early to preserve other leaves).
    
    We use Z-score normalization to make the heuristic robust to graph density.
    We weight the mean neighbor degree term more heavily (2.0) than the degree term (1.0),
    based on findings that this better prioritizes blocking high-degree hubs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = adjacency.shape[0]
    
    # Degrees
    deg = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of neighbor degrees
    # For vertex v, sum(deg[u] for u in N(v))
    neigh_deg_sum = adjacency.dot(deg)
    
    # Mean neighbor degree
    # Handle isolated nodes (deg=0) to avoid division by zero.
    # Isolated nodes have no neighbors, so their "mean neighbor degree" is 0.
    mean_neigh_deg = np.zeros(n)
    mask = deg > 0
    mean_neigh_deg[mask] = neigh_deg_sum[mask] / deg[mask]
    
    # Z-score normalization
    def zscore(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return np.zeros_like(x)
        return (x - mean) / std
        
    z_deg = zscore(deg)
    z_mean_neigh = zscore(mean_neigh_deg)
    
    # Check time (unlikely to be needed for N=200, but good practice)
    if time.time() > deadline:
        # Fallback: simple low degree priority
        return -deg
        
    # Weights:
    # w1 for degree (negative because we want low degree)
    # w2 for mean neighbor degree (positive because we want high neighbor degree)
    # Finding suggests w2 > w1 is beneficial.
    w1 = 1.0
    w2 = 2.0
    
    scores = -w1 * z_deg + w2 * z_mean_neigh
    
    return scores
