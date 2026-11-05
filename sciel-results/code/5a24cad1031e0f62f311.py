import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # Helper to calculate the score (max degree sum over triangles)
    def calc_score(adj):
        degs = np.sum(adj, axis=1)
        max_sum = 0
        # Iterate over edges
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Common neighbors
                    common = np.where(adj[u] & adj[v] > 0)[0]
                    for w in common:
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    # Initialize with K_{8,8} + 1 edge
    adj = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Add one edge within Part A to reach 65 edges (64 + 1)
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_score = calc_score(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing with Edge Swaps
    # Move: Remove a random edge (u,v), Add a random non-edge (x,y)
    # This maintains exactly 65 edges.
    
    T = 5.0
    cooling = 0.9995
    iterations = 200000
    
    # Precompute list of edges and non-edges for faster sampling?
    # Or just sample vertices.
    
    for i in range(iterations):
        # 1. Pick a random edge to remove
        # We need to pick (u,v) such that adj[u,v] == 1
        # Simple way: sample u, v, check if edge exists. If not, retry.
        # To make it faster, we can maintain a list of edges.
        # But n=16, so sampling is fast.
        
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        while u == v or adj[u, v] == 0:
            u = rng.integers(0, n)
            v = rng.integers(0, n)
        
        # Ensure u < v for consistency? Not strictly necessary for removal.
        
        # 2. Pick a random non-edge to add
        x = rng.integers(0, n)
        y = rng.integers(0, n)
        while x == y or adj[x, y] == 1:
            x = rng.integers(0, n)
            y = rng.integers(0, n)
            
        # Ensure we don't remove and add the same edge (if we just picked it)
        # If {u,v} == {x,y}, the move is identity.
        if (u == x and v == y) or (u == y and v == x):
            continue

        # Apply move
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        new_score = calc_score(adj)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Revert
            adj[u, v] = 1
            adj[v, u] = 1
            adj[x, y] = 0
            adj[y, x] = 0
            
        T *= cooling
        
        if best_score <= 23:
            # If we hit the conjectured asymptotic bound, we can try to push for 22
            # But 23 is already very good.
            pass

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
