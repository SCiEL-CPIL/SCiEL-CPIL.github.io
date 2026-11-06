import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 16
    target_edges = 65
    
    # Helper: build adjacency masks from edge list
    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def edge_set(adj):
        s = set()
        for u in range(n):
            m = adj[u]
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                if v > u:
                    s.add((u, v))
        return s
    
    def compute_score(adj):
        """Compute max triangle degree-sum using bitmasks."""
        degs = [bin(adj[u]).count('1') for u in range(n)]
        max_score = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        w = (common & -common).bit_length() - 1
                        common &= common - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_score:
                            max_score = s
        return max_score
    
    # Start from K_{6,10} + 5 internal edges
    # Part A: vertices 0-5, Part B: vertices 6-15
    # Cross edges: all 6*10 = 60
    # Need 5 more edges internal to part B (10 vertices, choose 5)
    edges = []
    for u in range(6):
        for v in range(6, 16):
            edges.append([u, v])
    # 5 internal edges in part B: pick 5 edges from the 10 vertices
    # Use a matching-like structure to minimize degree increases
    internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in internal_edges:
        edges.append([u, v])
    
    adj = build_adj(edges)
    best_score = compute_score(adj)
    best_adj = adj[:]
    
    # Local search: edge swaps
    rng = random.Random(42)
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # For speed, precompute current edge set
    current_edge_set = edge_set(current_adj)
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try random edge swaps
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            e1 = rng.choice(list(current_edge_set))
            # Pick a random edge to add (not already present)
            e2 = rng.choice(all_edges)
            if e2 in current_edge_set:
                continue
            
            # Apply swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Check validity: e2 not in current, e1 in current
            # Remove e1
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_edge_set.discard(e1)
            
            # Add e2
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            current_edge_set.add(e2)
            
            new_score = compute_score(current_adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
                improved = True
            else:
                # Revert
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                current_edge_set.discard(e2)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_edge_set.add(e1)
    
    # Return best result
    best_edge_set = edge_set(best_adj)
    best_edges = [[u, v] for u, v in best_edge_set]
    
    return {'n': n, 'edges': best_edges}
