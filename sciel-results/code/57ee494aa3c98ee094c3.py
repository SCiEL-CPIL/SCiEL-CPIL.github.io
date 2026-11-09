"""Rewritten program: core-decomposition-based priority for Maximum Independent Set."""
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex. The greedy decoder selects vertices
    in descending score order (i.e., most positive first), skipping neighbors
    of already-selected vertices.
    
    Strategy: Use core decomposition (peeling). Vertices removed early (low core
    number) are peripheral and good candidates for the independent set. We assign
    higher scores to vertices with lower core numbers. Ties are broken by lower
    original degree.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=float)
    
    deadline = time.time() + 100.0  # safety margin under 120s
    
    # Convert to a workable form
    adj = np.asarray(adjacency, dtype=bool)
    if adj.shape != (n, n):
        # Fallback: just use degree
        deg = adj.sum(axis=1).astype(float)
        return -deg
    
    # Core decomposition via peeling
    # current_deg: current degree of each vertex
    # alive: whether vertex is still in the graph
    current_deg = adj.sum(axis=1).astype(np.int64).copy()
    alive = np.ones(n, dtype=bool)
    core_layer = np.zeros(n, dtype=np.int64)
    
    # For efficiency with N=200, we can do a simple loop
    # At each step, find the alive vertex with minimum current degree
    # Remove it, assign it a layer, and update degrees of its neighbors
    
    remaining = n
    while remaining > 0:
        if time.time() > deadline:
            # Fallback: use what we have, assign remaining vertices a high layer
            for i in range(n):
                if alive[i]:
                    core_layer[i] = remaining  # high layer = low priority
            break
        
        # Find alive vertex with minimum current degree
        # For N=200, a linear scan is fine
        min_deg = n + 1
        min_v = -1
        for i in range(n):
            if alive[i] and current_deg[i] < min_deg:
                min_deg = current_deg[i]
                min_v = i
        
        if min_v == -1:
            break
        
        # Determine the layer: all vertices with this same minimum degree
        # in this "round" get the same layer number
        layer = min_deg
        
        # Find all alive vertices with current degree == min_deg
        # Remove them all in this round
        to_remove = []
        for i in range(n):
            if alive[i] and current_deg[i] == min_deg:
                to_remove.append(i)
        
        for v in to_remove:
            core_layer[v] = layer
            alive[v] = False
            remaining -= 1
            # Update degrees of neighbors
            neighbors = np.where(adj[v, :])[0]
            for nb in neighbors:
                if alive[nb]:
                    current_deg[nb] -= 1
    
    # Build scores:
    # Higher score = higher priority = selected first
    # We want low core_layer to have high score
    # Tie-break: lower original degree should have slightly higher score
    orig_deg = adj.sum(axis=1).astype(float)
    
    # Normalize core_layer to [0, 1] range
    max_layer = core_layer.max() if n > 0 else 1
    if max_layer == 0:
        max_layer = 1
    
    # Score = -(core_layer / max_layer) * (1 + orig_deg / n)
    # This makes low core_layer dominate, with degree as a small tie-breaker
    # Lower value of this expression = higher priority (since decoder picks descending)
    # Wait: decoder picks in DESCENDING score order. So we want high-priority vertices
    # to have HIGH scores.
    
    # Let's invert: score = (max_layer - core_layer) / max_layer * base_weight - orig_deg / n * tie_weight
    # Simpler: score = -core_layer * 1.0 - orig_deg * 0.01
    # Vertices with low core_layer and low degree get higher (less negative) scores
    
    score = -core_layer.astype(float) * 1.0 - orig_deg * 0.01
    
    return score
