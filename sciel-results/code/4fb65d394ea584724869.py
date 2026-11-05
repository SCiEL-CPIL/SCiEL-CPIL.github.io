import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    # Degrees
    deg = adjacency.sum(axis=1).astype(float)
    
    # Avoid division by zero for isolated nodes
    # Isolated nodes have degree 0. They don't block anyone.
    # Their neighbor degree is undefined.
    # We should give them high priority? Or low?
    # If deg=0, they can always be picked.
    # In -deg + mean_neigh_deg:
    # If deg=0, score = 0 + 0 = 0.
    # A node with deg=1, neigh deg=100 -> score = -1 + 100 = 99.
    # So isolated nodes (0) are picked after "good" nodes (99) but before "bad" nodes (negative).
    # This seems reasonable. Isolated nodes are free picks.
    
    # Calculate sum of neighbor degrees
    neigh_deg_sum = adjacency.dot(deg)
    
    # Mean neighbor degree
    # Use np.where to handle deg=0
    mean_neigh_deg = np.zeros(n)
    mask = deg > 0
    mean_neigh_deg[mask] = neigh_deg_sum[mask] / deg[mask]
    
    # Z-score normalization
    # Standard deviation might be 0 for uniform graphs?
    # Erdos-Renyi graphs are random, so variance should be non-zero.
    # But handle edge cases.
    
    def zscore(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return np.zeros_like(x)
        return (x - mean) / std
        
    z_deg = zscore(deg)
    z_mean_neigh = zscore(mean_neigh_deg)
    
    # Score: -degree (want low deg) + mean_neighbor_degree (want high neigh deg)
    # Weights: w1 for degree, w2 for mean neighbor degree.
    # Seed was effectively: -1 * deg + 0 * mean_neigh (unnormalized)
    # Let's try equal weights on normalized features.
    w1 = 1.0
    w2 = 1.0
    
    scores = -w1 * z_deg + w2 * z_mean_neigh
    
    return scores
