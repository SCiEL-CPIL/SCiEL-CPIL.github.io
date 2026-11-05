"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_k_core(adjacency, n):
    """Compute k-core numbers for each vertex using iterative peeling."""
    deg = adjacency.sum(axis=1).astype(int).copy()
    k_core = np.zeros(n, dtype=np.int32)
    
    if n == 0:
        return k_core
    
    # Use a queue-based peeling algorithm
    # Start with all vertices, peel those with degree < current k
    # For efficiency, we'll do iterative removal
    
    active = np.ones(n, dtype=bool)
    current_deg = deg.copy()
    max_k = int(deg.max()) if n > 0 else 0
    
    # For each vertex, its k-core number is the largest k such that it's in the k-core
    # We peel from k=1 upward
    
    # Simple O(N^2) approach for N=200
    # For k from 1 to max_k:
    #   Remove all vertices with current degree < k
    #   Repeat until stable
    #   Vertices remaining after peeling for k have k-core >= k
    
    # More efficient: process vertices in order of degree
    # Standard algorithm:
    # 1. Sort vertices by degree
    # 2. Use a queue of vertices with degree < current threshold
    
    # Let's use the standard peeling:
    # Initialize: all vertices active
    # For k = 1, 2, 3, ...:
    #   While there exists a vertex v with current_deg[v] < k:
    #     Mark v as having k_core = k-1 (if not already set)
    #     Remove v, decrease neighbors' degrees
    
    # Actually, the standard way:
    # k_core[v] = max{k : v is in the k-core}
    # 
    # Algorithm:
    # 1. Let d[v] = degree of v
    # 2. Sort vertices by d[v]
    # 3. Process in order, maintaining a count of how many neighbors have been "removed" with lower core
    
    # Let me use a simpler iterative approach for N=200
    k_core = np.zeros(n, dtype=np.int32)
    remaining_deg = deg.copy()
    removed = np.zeros(n, dtype=bool)
    
    # For k from 1 to max_k:
    #   Find all vertices not removed with remaining_deg < k
    #   Set their k_core to k-1 (if not set yet) and remove them
    #   Update neighbors' remaining_deg
    #   Repeat until no more vertices to remove for this k
    # After all k, remaining vertices have k_core = max_k (or their actual core)
    
    # Actually, let me do it properly:
    # The k-core of a vertex is determined by the peeling process.
    # We peel vertices in order of increasing degree.
    
    # Use a bucket sort / counting sort approach
    # For N=200, simple O(N^2) is fine
    
    active = np.ones(n, dtype=bool)
    cur_deg = deg.copy()
    
    # We'll determine k_core by peeling
    # Start with k=1
    # For k in range(1, max_k+1):
    #   changed = True
    #   while changed:
    #     changed = False
    #     for v in range(n):
    #       if active[v] and cur_deg[v] < k:
    #         k_core[v] = k - 1
    #         active[v] = False
    #         changed = True
    #         for u in range(n):
    #           if active[u] and adjacency[u, v]:
    #             cur_deg[u] -= 1
    #   # End while
    # End for
    # For remaining active vertices, k_core = max_k (or the last k tried)
    
    # This is O(max_k * N * E) worst case, but for N=200 it's fine
    # Let me optimize slightly
    
    for k in range(1, max_k + 1):
        changed = True
        while changed:
            changed = False
            # Find vertices to remove
            to_remove = []
            for v in range(n):
                if active[v] and cur_deg[v] < k:
                    to_remove.append(v)
            if not to_remove:
                break
            for v in to_remove:
                if active[v]:  # might have been removed already in this pass
                    k_core[v] = k - 1
                    active[v] = False
                    changed = True
                    # Update neighbors
                    neighbors = np.where(adjacency[v])[0]
                    for u in neighbors:
                        if active[u]:
                            cur_deg[u] -= 1
    
    # Remaining active vertices have k_core = max_k
    # But actually, they might have higher core... but max_k is the max possible
    # Set remaining to max_k
    for v in range(n):
        if active[v]:
            k_core[v] = max_k
    
    return k_core


def priority(adjacency):
    """
    Priority score for greedy independent set.
    Higher score = picked earlier.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    deg = adjacency.sum(axis=1).astype(float)
    
    # Compute k-core numbers
    k_core = _compute_k_core(adjacency, n).astype(float)
    
    # Mean degree of neighbors
    # For each vertex v, mean degree of its neighbors
    # If isolated, use 0
    neighbor_deg_sum = adjacency.astype(float) @ deg
    neighbor_count = deg
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(neighbor_count > 0, neighbor_deg_sum / np.maximum(neighbor_count, 1), 0.0)
    
    # Strategy: prioritize vertices with LOW k-core (sparse regions)
    # and LOW degree
    # Also, vertices whose neighbors have HIGH mean degree are good to pick
    # (because those neighbors are in dense areas and less likely to be in MIS)
    
    # Combine: score = -k_core * w1 - deg * w2 + mean_neighbor_deg * w3
    # Let's try: primarily k_core, then degree
    # Lower k_core is better, lower degree is better
    # But we want HIGH score to be picked first
    
    # score = -k_core - 0.1 * deg + 0.05 * mean_neighbor_deg
    # Or simpler: just -k_core - deg
    
    # Let's use a weighted combination
    # Normalize each component to [0,1] range for stable weighting
    k_core_max = k_core.max() if k_core.max() > 0 else 1.0
    deg_max = deg.max() if deg.max() > 0 else 1.0
    mnd_max = mean_neighbor_deg.max() if mean_neighbor_deg.max() > 0 else 1.0
    
    k_core_norm = k_core / k_core_max
    deg_norm = deg / deg_max
    mnd_norm = mean_neighbor_deg / mnd_max
    
    # We want low k_core and low degree to have high priority
    # High mean_neighbor_deg is slightly good
    score = -k_core_norm * 1.0 - deg_norm * 0.5 + mnd_norm * 0.1
    
    return score
# EVOLVE-BLOCK-END
