"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Use a multi-faceted heuristic that combines:
    1. Core number (from k-core decomposition) - vertices with lower core numbers
       are "peripheral" and safer to pick early without blocking many others.
    2. Degree - lower degree vertices are better candidates.
    3. A "soft" measure of how many neighbors have high core numbers, which helps
       identify vertices that can be picked while their high-core neighbors remain
       in the graph (they won't be picked early anyway).
    
    The greedy decoder picks in descending score order, so we want higher scores
    for vertices that are "safe" to pick early (low core, low degree, surrounded
    by high-core vertices that will be skipped).
    """
    n = adjacency.shape[0]
    
    # Deadline guard: if we take too long, fall back to simple degree-based score
    start_time = time.time()
    DEADLINE = 100  # seconds, well under the 150s limit
    
    # Compute degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Compute core numbers via peeling process
    # Core number of a vertex is the largest k such that the vertex is in the k-core
    core_numbers = np.zeros(n, dtype=int)
    deg = degrees.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    
    # Efficient k-core peeling
    # Use a bucket queue approach
    from collections import deque
    
    # Initialize buckets
    max_deg = int(deg.max()) if n > 0 else 0
    buckets = [deque() for _ in range(max_deg + 1)]
    for i in range(n):
        buckets[deg[i]].append(i)
    
    # Peeling process
    core = np.zeros(n, dtype=int)
    k = 0
    for k in range(max_deg + 1):
        # Process all vertices with current degree k
        while buckets[k] and remaining[buckets[k][0]]:
            v = buckets[k].popleft()
            if not remaining[v]:
                continue
            core[v] = k
            remaining[v] = False
            # Remove neighbors
            for u in np.where(adjacency[v])[0]:
                if remaining[u]:
                    deg[u] -= 1
                    if deg[u] < k:
                        buckets[deg[u]].append(u)
                    elif deg[u] == k:
                        # Already in current bucket or will be processed
                        pass
    
    # After peeling, remaining vertices have core number >= max_deg
    # Actually, let me redo this more carefully using the standard algorithm
    
    # Standard k-core algorithm
    core_numbers = np.zeros(n, dtype=int)
    if n == 0:
        return np.array([], dtype=float)
    
    deg = adjacency.sum(axis=1).astype(int)
    remaining = np.ones(n, dtype=bool)
    remaining_count = n
    
    # Use iterative peeling
    # Find min degree repeatedly
    # For efficiency with n=200, even O(n^2) is fine
    for k in range(1, int(deg.max()) + 1):
        # Find all vertices with degree < k that are still remaining
        # and remove them iteratively
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if remaining[i] and deg[i] < k:
                    remaining[i] = False
                    core_numbers[i] = k - 1
                    changed = True
                    # Update neighbors
                    for j in np.where(adjacency[i])[0]:
                        if remaining[j]:
                            deg[j] -= 1
        # If we couldn't remove any more, all remaining have degree >= k
    
    # Vertices that were never removed have core number = max core
    max_core = int(core_numbers.max()) if n > 0 else 0
    for i in range(n):
        if remaining[i]:
            core_numbers[i] = max_core
    
    # Check deadline
    if time.time() - start_time > DEADLINE * 0.5:
        # Fall back to simple score
        return -degrees
    
    # Compute neighbor core number statistics
    # For each vertex, compute the mean core number of its neighbors
    neighbor_core_sums = adjacency.dot(core_numbers.astype(float))
    neighbor_counts = degrees
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_core = np.where(neighbor_counts > 0, neighbor_core_sums / neighbor_counts, 0)
    
    # Also compute: how many neighbors have higher core number than this vertex
    # This indicates "support" from the core
    higher_core_neighbor_counts = np.zeros(n, dtype=float)
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) > 0:
            higher_core_neighbor_counts[i] = np.sum(core_numbers[neighbors] > core_numbers[i])
    
    # Score design:
    # We want to pick vertices that:
    # 1. Have low core number (peripheral, won't block core structure)
    # 2. Have low degree (fewer neighbors to block)
    # 3. Have many neighbors with HIGHER core number (those neighbors will be 
    #    skipped by the greedy decoder anyway since they have lower priority)
    #
    # The greedy decoder picks in DESCENDING score order. So high score = picked first.
    # We want peripheral, low-degree vertices to be picked first.
    #
    # Score = -alpha * degree - beta * core_number + gamma * higher_core_neighbors
    
    # Normalize to make weights comparable
    deg_norm = degrees / (degrees.max() + 1e-9)
    core_norm = core_numbers.astype(float) / (core_numbers.max() + 1e-9)
    hcn_norm = higher_core_neighbor_counts / (higher_core_neighbor_counts.max() + 1e-9)
    
    # Weighted combination
    alpha = 1.0
    beta = 0.5
    gamma = 0.8
    
    scores = -alpha * deg_norm - beta * core_norm + gamma * hcn_norm
    
    # Check deadline
    if time.time() - start_time > DEADLINE * 0.8:
        return scores
    
    # Add a small tie-breaking based on vertex index for determinism
    scores += np.arange(n) * 1e-10
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
