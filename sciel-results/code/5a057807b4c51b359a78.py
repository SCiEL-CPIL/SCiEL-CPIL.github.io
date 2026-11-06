import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    degrees = adjacency.sum(axis=1).astype(float)
    safe_degrees = np.maximum(degrees, 1.0)
    
    # PageRank iteration
    d = 0.85
    r = np.ones(n) / n
    
    for _ in range(100):
        # r_j / deg(j)
        contrib = r / safe_degrees
        # Sum contributions from neighbors
        neighbor_sum = adjacency.dot(contrib)
        # Dangling nodes (degree 0) contribution is handled by safe_degrees, 
        # but strictly PageRank distributes dangling mass uniformly.
        # In ER graphs, dangling nodes are rare.
        # r_new = (1-d)/N + d * neighbor_sum
        r_new = (1 - d) / n + d * neighbor_sum
        
        if np.linalg.norm(r_new - r) < 1e-6:
            break
        r = r_new
        
    # High PageRank = Central/Important.
    # We want to avoid important nodes.
    scores = -r
    
    return scores
