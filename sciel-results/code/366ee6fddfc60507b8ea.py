"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill
    
    n = 40
    
    # Start with 8 disjoint K5s (80 edges)
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    
    # Add a tree connecting the 8 K5s (7 edges) - safe, no C6 possible with 2-edge-connected blocks
    for k in range(7):
        # Connect representative vertex of block k to representative of block k+1
        # Block k has vertices 5k..5k+4, representative is 5k
        # To be safe, connect 5k to 5(k+1)
        edges.append([5 * k, 5 * (k + 1)])
    
    current_edges = set()
    for e in edges:
        u, v = e
        if u > v:
            u, v = v, u
        current_edges.add((u, v))
    
    best_edges = list(current_edges)
    best_count = len(best_edges)
    
    # Build adjacency sets for C6 checking
    adj = [set() for _ in range(n)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper: check if adding edge (u,v) creates a C6
    def creates_c6(u, v, adj):
        # A C6 through (u,v) would be u-v-a-b-c-d-u where a,b,c,d are distinct and not u,v
        # Equivalently, we need a path of length 4 from u to v that doesn't use the edge (u,v)
        # i.e., a path u-x-y-z-v where x,y,z distinct, not u,v, and edges (u,x),(x,y),(y,z),(z,v) exist
        
        # BFS/DFS for path of length 4 from u to v avoiding direct edge
        # Actually, we need to check if there's a path of exactly 4 edges from u to v
        # that doesn't go through (u,v) directly
        
        # Simple check: find all vertices at distance 1 from u (neighbors of u except v)
        # Then distance 2 from u, etc. We need v to be at distance exactly 4 from u
        # where the path doesn't use edge (u,v)
        
        # Since we're adding (u,v), the new edge creates a C6 if there's already a path of length 4 from u to v
        # Let's do a BFS from u avoiding v as first step... actually just check if dist(u,v) == 4 in current graph
        
        # BFS from u
        from collections import deque
        dist = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            node = q.popleft()
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        
        return dist[v] == 4
    
    # Try to add more edges between blocks
    # Generate candidate edges in some order
    all_possible = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_possible.add((i, j))
    
    candidates = list(all_possible - current_edges)
    random.shuffle(candidates)
    
    # Try adding edges one by one
    for u, v in candidates:
        if time.time() > deadline:
            break
        if (u, v) in current_edges or (v, u) in current_edges:
            continue
        if not creates_c6(u, v, adj):
            # Safe to add
            current_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            best_edges = list(current_edges)
            best_count = len(best_edges)
    
    # Local search: try to swap edges to increase count
    # For now, return what we have
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
