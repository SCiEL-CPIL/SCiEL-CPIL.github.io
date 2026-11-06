import time
import random
import numpy as np

def _count_common_neighbors(adj, u, v, w):
    # adj is a boolean numpy array
    # Count common neighbors of u, v, w
    # Exclude u, v, w themselves? 
    # In a simple graph, u is not a neighbor of u.
    # So adj[u] & adj[v] & adj[w] automatically excludes u,v,w if the diagonal is 0.
    # We ensure diagonal is 0.
    common = adj[u] & adj[v] & adj[w]
    return np.sum(common)

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u, v) keeps the graph K3,3-free.
    adj is a boolean numpy array of shape (n, n).
    """
    n = adj.shape[0]
    
    # We need to check if adding (u,v) creates a triple with 3 common neighbors.
    # The new common neighbors can only be u or v themselves, appearing in the neighborhood of other pairs.
    
    # Case 1: v becomes a common neighbor of (u, x, y)
    # This requires v to be in N(x) and N(y).
    # And u, x, y must already have at least 2 common neighbors.
    # If they have 2, adding v makes it 3.
    
    # Case 2: u becomes a common neighbor of (v, x, y)
    # This requires u to be in N(x) and N(y).
    # And v, x, y must already have at least 2 common neighbors.
    
    # Iterate over all pairs (x, y) distinct from u, v
    for x in range(n):
        if x == u or x == v:
            continue
        for y in range(x + 1, n):
            if y == u or y == v:
                continue
            
            # Check Case 1: v in N(x) and v in N(y)
            if adj[v, x] and adj[v, y]:
                # Check common neighbors of u, x, y
                # If count >= 2, adding v to N(u) makes it >= 3
                if _count_common_neighbors(adj, u, x, y) >= 2:
                    return False
            
            # Check Case 2: u in N(x) and u in N(y)
            if adj[u, x] and adj[u, y]:
                # Check common neighbors of v, x, y
                # If count >= 2, adding u to N(v) makes it >= 3
                if _count_common_neighbors(adj, v, x, y) >= 2:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Try multiple random restarts
    # We can also try different initial densities
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Randomize edge order
        edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges_to_try)
        
        # Start with empty graph
        adj = np.zeros((n, n), dtype=bool)
        
        current_count = 0
        current_edges = []
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if adj[u, v]:
                continue
            if _can_add_edge(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                current_count += 1
                current_edges.append([u, v])
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges[:]
            
        # If we find a very good solution, we might stop early or continue
        # Theoretical max is unknown, but > 36 is likely.
        # Let's run until deadline.
        
    return best_edges
