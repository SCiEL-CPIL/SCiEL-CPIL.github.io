"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign priority scores to vertices for greedy MIS.
    Higher score = selected earlier.
    
    Strategy: Combine structural features to favor vertices that are
    low-degree but adjacent to high-degree hubs (hub-blocking), with
    penalties for high clustering and high eigenvector centrality.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit with safety margin
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Degree
    degree = adj.sum(axis=1)
    
    # Neighbor degree sum (hub-blocking signal)
    # For each vertex, sum of degrees of its neighbors
    neighbor_degree_sum = adj @ degree
    
    # Clustering coefficient (approximate for efficiency)
    # For sparse graphs, compute local clustering
    # C(i) = 2 * (edges among neighbors) / (deg(i) * (deg(i)-1))
    clustering = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if time.time() > deadline - 5.0:
            break
        deg_i = int(degree[i])
        if deg_i < 2:
            clustering[i] = 0.0
            continue
        neighbors = np.where(adj[i] > 0)[0]
        # Count edges among neighbors
        sub_adj = adj[np.ix_(neighbors, neighbors)]
        edges_among = (sub_adj.sum() - np.trace(sub_adj)) / 2.0
        clustering[i] = 2.0 * edges_among / (deg_i * (deg_i - 1))
    
    # Eigenvector centrality (power iteration)
    eig_centrality = np.ones(n, dtype=np.float64)
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        new_eig = adj @ eig_centrality
        norm = np.linalg.norm(new_eig)
        if norm < 1e-12:
            break
        new_eig /= norm
        if np.linalg.norm(new_eig - eig_centrality) < 1e-8:
            eig_centrality = new_eig
            break
        eig_centrality = new_eig
    
    # Normalize features to [0, 1] range
    def normalize(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    deg_norm = normalize(degree)
    neighbor_deg_norm = normalize(neighbor_degree_sum)
    clustering_norm = normalize(clustering)
    eig_norm = normalize(eig_centrality)
    
    # Composite score:
    # We want HIGH score for vertices selected early.
    # - Low degree is good: (1 - deg_norm)
    # - High neighbor degree sum is good (hub-blocking): neighbor_deg_norm
    # - Low clustering is good: (1 - clustering_norm)
    # - Low eigenvector centrality is good: (1 - eig_norm)
    
    # Weighted combination
    w_deg = 0.35
    w_neighbor = 0.30
    w_cluster = 0.20
    w_eig = 0.15
    
    score = (
        w_deg * (1.0 - deg_norm) +
        w_neighbor * neighbor_deg_norm +
        w_cluster * (1.0 - clustering_norm) +
        w_eig * (1.0 - eig_norm)
    )
    
    # Add tiny random tie-breaker to ensure strict ordering
    rng = np.random.default_rng(42)
    score += rng.uniform(0, 1e-10, size=n)
    
    return score.astype(np.float64)
# EVOLVE-BLOCK-END
