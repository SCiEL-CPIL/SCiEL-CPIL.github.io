"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_k_core(adjacency, n):
    """Compute k-core numbers via iterative peeling."""
    deg = adjacency.sum(axis=1).astype(int)
    k_core = np.zeros(n, dtype=int)
    removed = np.zeros(n, dtype=bool)
    current_deg = deg.copy()
    
    # Use a simple iterative approach
    for k in range(1, n + 1):
        candidates = np.where((current_deg < k) & (~removed))[0]
        if len(candidates) == 0:
            break
        for v in candidates:
            if not removed[v]:
                removed[v] = True
                k_core[v] = k - 1
                # Decrease degree of neighbors
                neighbors = np.where(adjacency[v])[0]
                for u in neighbors:
                    if not removed[u]:
                        current_deg[u] -= 1
    
    # Any remaining vertices have k_core at least the max k we reached
    remaining = np.where(~removed)[0]
    if len(remaining) > 0:
        k_core[remaining] = k
    
    return k_core


def _compute_neighbor_avg_degree(adjacency, n):
    """Average degree of neighbors for each vertex."""
    deg = adjacency.sum(axis=1).astype(float)
    # For each vertex, average degree of its neighbors
    # adjacency @ deg gives sum of neighbor degrees
    neighbor_degree_sum = adjacency @ deg
    # Avoid division by zero
    safe_deg = np.where(deg > 0, deg, 1.0)
    neighbor_avg = neighbor_degree_sum / safe_deg
    # For isolated vertices, set to 0
    neighbor_avg[deg == 0] = 0.0
    return neighbor_avg


def _compute_clustering(adjacency, n):
    """Approximate local clustering coefficient for each vertex."""
    deg = adjacency.sum(axis=1).astype(int)
    clustering = np.zeros(n, dtype=float)
    
    # For efficiency, only compute for vertices with degree >= 2
    for v in range(n):
        d = deg[v]
        if d < 2:
            clustering[v] = 0.0
            continue
        # Get neighbors
        neighbors = np.where(adjacency[v])[0]
        # Count edges among neighbors
        sub_adj = adjacency[np.ix_(neighbors, neighbors)]
        edges_among = sub_adj.sum() / 2.0
        possible = d * (d - 1) / 2.0
        clustering[v] = edges_among / possible
    
    return clustering


def priority(adjacency):
    """
    Assign scores to vertices for greedy independent set selection.
    Higher score = more likely to be selected earlier.
    
    Strategy: Combine multiple structural features:
    - Low degree (prefer)
    - Low k-core number (prefer - sparse regions)
    - Low neighbor average degree (prefer)
    - Low clustering (prefer - less dense local structure)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    # Basic degree
    degree = adjacency.sum(axis=1).astype(float)
    
    # k-core numbers
    k_core = _compute_k_core(adjacency, n)
    k_core_float = k_core.astype(float)
    
    # Normalize features to [0, 1] range for consistent weighting
    def normalize(x):
        if x.max() == x.min():
            return np.zeros_like(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min())
    
    deg_norm = normalize(degree)
    kcore_norm = normalize(k_core_float)
    
    # Check time
    if time.time() > deadline:
        # Fallback: just use degree
        return -degree
    
    # Neighbor average degree
    neighbor_avg_deg = _compute_neighbor_avg_degree(adjacency, n)
    neighbor_avg_norm = normalize(neighbor_avg_deg)
    
    # Check time
    if time.time() > deadline:
        # Fallback: use degree and k-core
        return -(0.6 * deg_norm + 0.4 * kcore_norm)
    
    # Clustering coefficient
    clustering = _compute_clustering(adjacency, n)
    clustering_norm = normalize(clustering)
    
    # Check time
    if time.time() > deadline:
        # Fallback: use degree, k-core, neighbor avg
        return -(0.4 * deg_norm + 0.3 * kcore_norm + 0.3 * neighbor_avg_norm)
    
    # Second-order: average k-core of neighbors
    neighbor_kcore_avg = adjacency @ k_core_float
    safe_deg = np.where(degree > 0, degree, 1.0)
    neighbor_kcore_avg = neighbor_kcore_avg / safe_deg
    neighbor_kcore_avg[degree == 0] = 0.0
    neighbor_kcore_norm = normalize(neighbor_kcore_avg)
    
    # Check time
    if time.time() > deadline:
        return -(0.35 * deg_norm + 0.25 * kcore_norm + 0.2 * neighbor_avg_norm + 0.2 * clustering_norm)
    
    # Combined score: we want to SELECT vertices with LOW structural density
    # So we negate the combined "density" score
    # Weights tuned to emphasize k-core (sparse regions) and degree
    
    w_deg = 0.25
    w_kcore = 0.30
    w_neighbor_deg = 0.20
    w_clustering = 0.15
    w_neighbor_kcore = 0.10
    
    density_score = (
        w_deg * deg_norm +
        w_kcore * kcore_norm +
        w_neighbor_deg * neighbor_avg_norm +
        w_clustering * clustering_norm +
        w_neighbor_kcore * neighbor_kcore_norm
    )
    
    # Negate so that lower density = higher priority
    scores = -density_score
    
    # Add small tie-breaker based on vertex index for determinism
    # This helps ensure consistent behavior
    scores += np.arange(n, dtype=float) * 1e-10
    
    return scores
# EVOLVE-BLOCK-END
