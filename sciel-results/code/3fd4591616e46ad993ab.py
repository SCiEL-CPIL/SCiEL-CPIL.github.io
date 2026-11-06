import random
import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Leave 5 seconds buffer
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_arr = np.array(all_pairs, dtype=np.int32)
    
    best_edges = []
    best_count = 0
    
    def is_valid_addition(adj, codeg, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        # Check codeg(u, v) <= 2
        if codeg[u, v] > 2:
            return False
        
        # For each neighbor x of v (x != u), check codeg(u, x) <= 1
        # Because adding (u,v) makes v a common neighbor of u and x if x is neighbor of v
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v, x]:
                if codeg[u, x] > 1:
                    return False
        
        # For each neighbor y of u (y != v), check codeg(v, y) <= 1
        for y in range(n):
            if y == u or y == v:
                continue
            if adj[u, y]:
                if codeg[v, y] > 1:
                    return False
        
        return True
    
    def update_codeg(adj, codeg, u, v):
        """Update codeg matrix after adding edge (u, v)."""
        # When adding (u, v):
        # v becomes a new common neighbor of u and x for each x in N(v) \ {u}
        # u becomes a new common neighbor of v and y for each y in N(u) \ {v}
        
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v, x]:
                codeg[u, x] += 1
                codeg[x, u] += 1
        
        for y in range(n):
            if y == u or y == v:
                continue
            if adj[u, y]:
                codeg[v, y] += 1
                codeg[y, v] += 1
    
    # Greedy with randomized restarts
    # Try different strategies:
    # 1. Random order
    # 2. Degree-based priority
    # 3. Mixed approach
    
    restarts = 0
    while time.time() - start_time < time_limit:
        adj = np.zeros((n, n), dtype=np.int32)
        codeg = np.zeros((n, n), dtype=np.int32)
        edges = []
        
        # Create a shuffled list of pairs
        shuffled_pairs = all_pairs_arr.copy()
        np.random.shuffle(shuffled_pairs)
        
        # Try adding edges in random order
        for u, v in shuffled_pairs:
            if is_valid_addition(adj, codeg, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                update_codeg(adj, codeg, u, v)
                edges.append([int(u), int(v)])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            restarts += 1
        
        # If we've been running for a while, try a more sophisticated approach
        if restarts > 100 and time.time() - start_time < time_limit - 10:
            # Local search: start from best_edges and try to improve
            adj = np.zeros((n, n), dtype=np.int32)
            codeg = np.zeros((n, n), dtype=np.int32)
            current_edges = []
            
            for e in best_edges:
                u, v = e
                adj[u, v] = 1
                adj[v, u] = 1
                update_codeg(adj, codeg, u, v)
                current_edges.append([u, v])
            
            # Try to add more edges
            improved = True
            while improved and time.time() - start_time < time_limit:
                improved = False
                for u, v in all_pairs:
                    if adj[u, v] == 0:
                        if is_valid_addition(adj, codeg, u, v):
                            adj[u, v] = 1
                            adj[v, u] = 1
                            update_codeg(adj, codeg, u, v)
                            current_edges.append([u, v])
                            improved = True
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges.copy()
    
    return best_edges
