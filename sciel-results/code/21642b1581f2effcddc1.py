import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Uses an iterative k-core decomposition-like heuristic to assign scores.
    Vertices with lower "coreness" (i.e., those that would be removed earlier in
    a k-core peeling process) are prioritized. This effectively identifies the 
    "periphery" of the graph, which tends to contain more vertices suitable for 
    an independent set compared to the dense "core".
    
    The score is the negative of the core number (or a related peeling order), 
    so lower core numbers get higher priority.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Safety check for empty or trivial graphs
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])
    
    # Initialize degrees
    deg = adj.sum(axis=1).astype(int)
    
    # We will simulate the peeling process to assign a "core number" to each vertex.
    # The core number of a vertex is the largest k for which it belongs to the k-core.
    # In the context of MIS, vertices with lower core numbers are generally "easier" to include
    # because they are less constrained by high-density subgraphs.
    
    # Efficient k-core decomposition algorithm:
    # 1. Sort vertices by degree.
    # 2. Repeatedly remove the vertex with the lowest degree, updating neighbors.
    # 3. The order of removal (or the degree at removal) gives the core number.
    
    # To be fast, we can use a bucket sort approach or simply iterate.
    # Given n=200, a simple iterative removal is fast enough.
    
    # Copy degrees to track current degree
    current_deg = deg.copy()
    core_number = np.zeros(n, dtype=float)
    
    # We will process vertices in order of increasing current degree.
    # This is similar to the algorithm for k-core decomposition.
    
    # List of vertices not yet processed
    unprocessed = list(range(n))
    
    # For efficiency, we can maintain a min-heap or just scan for min.
    # With n=200, scanning is O(n^2) = 40,000 ops, which is trivial.
    
    processed = [False] * n
    
    while unprocessed:
        # Find vertex with minimum current degree among unprocessed
        min_idx = -1
        min_deg = float('inf')
        for v in unprocessed:
            if current_deg[v] < min_deg:
                min_deg = current_deg[v]
                min_idx = v
        
        if min_idx == -1:
            break
            
        # Assign core number (the degree at the time of removal)
        core_number[min_idx] = min_deg
        
        # Remove this vertex: update degrees of its neighbors
        neighbors = np.where(adj[min_idx] == 1)[0]
        for u in neighbors:
            if not processed[u]:
                current_deg[u] -= 1
        
        processed[min_idx] = True
        unprocessed.remove(min_idx)
        
        # Time check
        if time.time() - start_time > 140:
            # Fallback: just use degree
            return -deg

    # The score should favor vertices with LOW core numbers.
    # So, score = -core_number
    # However, core numbers can be tied. We can break ties by original degree (lower is better).
    # Let's combine: score = -core_number - 0.1 * deg
    
    score = -core_number - 0.1 * deg
    
    return score
