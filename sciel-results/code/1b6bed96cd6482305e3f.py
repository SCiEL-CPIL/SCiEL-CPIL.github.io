import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 40
    target_edges = 401
    
    # Represent graph as a 40x40 boolean adjacency matrix for fast lookups
    # adj[u][v] is True if edge (u,v) exists
    adj = np.zeros((n, n), dtype=bool)
    
    # Helper: count common neighbors (book size) for edge (u,v)
    # Uses vectorized numpy operations for speed
    def book_size(u, v):
        # Common neighbors of u and v
        # adj[u] & adj[v] gives a vector where both are True
        # Exclude u and v themselves (they can't be neighbors of themselves in a simple graph)
        common = adj[u] & adj[v]
        common[u] = False
        common[v] = False
        return int(np.sum(common))
    
    # Helper: compute maximum book size over all edges
    def max_book(adj_mat):
        max_b = 0
        # Iterate over upper triangle
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mat[u, v]:
                    b = int(np.sum(adj_mat[u] & adj_mat[v]))
                    if b > max_b:
                        max_b = b
        return max_b
    
    # Initialize with complete bipartite graph K_20,20 plus one extra edge
    # This is a good starting point because Edwards-Khadzhiivanov-Nikiforov lower bound
    # suggests graphs near the extremal construction.
    
    # First 20 vertices in set A, last 20 in set B
    for u in range(20):
        for v in range(20, 40):
            adj[u, v] = True
            adj[v, u] = True
    
    # Add one extra edge within set A
    adj[0, 1] = True
    adj[1, 0] = True
    
    current_edges_count = target_edges  # 400 + 1 = 401
    current_score = max_book(adj)
    
    best_adj = adj.copy()
    best_score = current_score
    
    # List of all possible edges for sampling
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Track which edges are present for fast sampling of absent edges
    present_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                present_edges.add((u, v))
    
    # Local search with swaps: remove one edge, add another (keeps count at 401)
    T = 2.0
    T_min = 0.1
    alpha = 0.9995
    
    while time.time() < deadline:
        # Sample a random absent edge to add
        # Since 401/780 ~ 51.4% are present, we can sample and retry
        new_u, new_v = random.sample(range(n), 2)
        if new_u > new_v:
            new_u, new_v = new_v, new_u
        new_edge = (new_u, new_v)
        
        attempts = 0
        while new_edge in present_edges and attempts < 20:
            new_u, new_v = random.sample(range(n), 2)
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            new_edge = (new_u, new_v)
            attempts += 1
        
        if new_edge in present_edges:
            continue
        
        # Sample a random present edge to remove
        # Build a list of present edges for O(1) random selection
        # We maintain present_edges as a set; convert to list for random.choice
        # To avoid rebuilding list every time, we can use a different approach:
        # Pick random u, v until we hit a present edge
        old_u, old_v = random.sample(range(n), 2)
        if old_u > old_v:
            old_u, old_v = old_v, old_u
        old_edge = (old_u, old_v)
        
        attempts = 0
        while old_edge not in present_edges and attempts < 20:
            old_u, old_v = random.sample(range(n), 2)
            if old_u > old_v:
                old_u, old_v = old_v, old_u
            old_edge = (old_u, old_v)
            attempts += 1
        
        if old_edge not in present_edges:
            continue
        
        # Apply move: remove old_edge, add new_edge
        adj[old_u, old_v] = False
        adj[old_v, old_u] = False
        adj[new_u, new_v] = True
        adj[new_v, new_u] = True
        
        present_edges.discard(old_edge)
        present_edges.add(new_edge)
        
        new_score = max_book(adj)
        delta = new_score - current_score
        
        if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Revert
            adj[old_u, old_v] = True
            adj[old_v, old_u] = True
            adj[new_u, new_v] = False
            adj[new_v, new_u] = False
            
            present_edges.discard(new_edge)
            present_edges.add(old_edge)
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Return best found as list of [u, v]
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                result.append([u, v])
    
    return result
