import numpy as np
import heapq

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Get initial degrees
    degrees = adjacency.sum(axis=1).astype(int)
    original_degrees = degrees.copy()
    
    # Prepare for peeling
    # We need to efficiently get neighbors. 
    # Since N is small, we can just use the matrix.
    # neighbors[i] = np.where(adjacency[i])[0]
    
    # Min-heap of (degree, vertex)
    heap = [(degrees[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removed = np.zeros(n, dtype=bool)
    removal_order = np.zeros(n, dtype=int)
    step = 0
    
    # Lazy deletion: we push updates to heap, but check if vertex is already removed or if degree in heap matches current degree
    # Actually, simpler: just push (new_degree, vertex). When popping, if removed[vertex] or degree != current_degree, skip.
    # But we need current_degree array.
    current_degrees = degrees.copy()
    
    while heap:
        d, u = heapq.heappop(heap)
        if removed[u]:
            continue
        if d != current_degrees[u]:
            # This entry is stale
            continue
            
        # Remove u
        removed[u] = True
        removal_order[u] = step
        step += 1
        
        # Update neighbors
        # Find neighbors of u
        # np.where(adjacency[u])[0] is O(N)
        neighbors = np.where(adjacency[u])[0]
        for v in neighbors:
            if not removed[v]:
                current_degrees[v] -= 1
                heapq.heappush(heap, (current_degrees[v], v))
                
    # Construct scores
    # We want low removal order to have high priority.
    # We want low degree to have high priority (tie breaker).
    # score = -removal_order * (n + 1) - original_degrees
    # This ensures order 0 > order 1, etc.
    scores = -removal_order.astype(float) * (n + 1) - original_degrees.astype(float)
    
    return scores
