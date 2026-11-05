# EVOLVE-BLOCK-START
import numpy as np
import heapq


def priority(adjacency):
    """
    Computes a priority score for each vertex using K-Core decomposition.
    
    Approach:
    1. Compute the core number for each vertex using a peeling algorithm.
       The core number represents the density of the local subgraph containing the vertex.
    2. Vertices with lower core numbers are more "peripheral" and are generally
       better candidates for an independent set because they constrain fewer
       other vertices and are less likely to be part of a dense core that
       requires excluding many vertices.
    3. Score is inversely proportional to the core number and degree.
       Higher score = Lower core number and Lower degree.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    N = adjacency.shape[0]
    
    # Compute degrees
    deg = adjacency.sum(axis=1).astype(int)
    
    # Pre-compute neighbor lists for efficient peeling
    neighbors = [np.where(adjacency[i] == 1)[0] for i in range(N)]
    
    # Initialize core numbers
    core = np.zeros(N, dtype=int)
    
    # Min-heap of (current_degree, vertex_index)
    # We need to handle stale entries in the heap
    heap = [(deg[i], i) for i in range(N)]
    heapq.heapify(heap)
    
    # Current degrees (will be modified)
    cur_deg = deg.copy()
    
    processed = np.zeros(N, dtype=bool)
    
    while heap:
        d, v = heapq.heappop(heap)
        
        if processed[v]:
            continue
            
        # If the degree in the heap is not the current degree, it's stale
        if d != cur_deg[v]:
            continue
        
        # Assign core number
        core[v] = d
        processed[v] = True
        
        # Update neighbors
        for u in neighbors[v]:
            if not processed[u]:
                cur_deg[u] -= 1
                heapq.heappush(heap, (cur_deg[u], u))
                
    # Create scores
    # We want to pick low core number and low degree first.
    # Score = 1 / (core + deg + 1)
    # If core is 0, it's a leaf or isolated. Very high priority.
    # If core is high, it's in a dense part. Low priority.
    
    # To ensure distinct scores for better greedy behavior (avoiding arbitrary tie-breaking),
    # we add a small epsilon based on index or use a combined metric.
    # However, the greedy decoder handles ties by index or stable sort.
    # We want to make sure that among same core, lower degree is preferred.
    
    # Normalize core and deg to [0, 1] or similar?
    # Simple linear combination:
    # score = - (core * 10 + deg)
    # This makes core the primary factor.
    
    # Let's use:
    # score = 1.0 / (core + 1.0) * (1.0 / (deg + 1.0))
    
    scores = 1.0 / (core + 1.0) * 1.0 / (deg + 1.0)
    
    return scores
# EVOLVE-BLOCK-END
