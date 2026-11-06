import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 40
    best_edges = 0
    best_adj = [0] * n  # List of bitmasks
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper function to check if adding (u,v) creates a C6
    # A C6 is formed if there is a path of length 5 between u and v in the current graph.
    def creates_c6(adj, u, v):
        # BFS using bitmasks
        # visited[i] is bitmask of nodes at distance i
        visited = [0] * 6
        visited[0] = 1 << u
        
        # We need to expand up to distance 5
        # dist[k] is the set of nodes at distance exactly k
        # dist[0] = {u}
        # dist[1] = N(u) \ {u}
        # dist[k] = (Union N(x) for x in dist[k-1]) \ (Union dist[0..k-1])
        
        dist = [0] * 6
        dist[0] = 1 << u
        visited_mask = 1 << u
        
        for k in range(1, 6):
            # Calculate neighbors of all nodes in dist[k-1]
            # We can iterate through set bits in dist[k-1]
            current_dist = dist[k-1]
            next_dist = 0
            # Optimization: if current_dist is 0, break
            while current_dist:
                # Get lowest set bit
                lsb = current_dist & -current_dist
                # Index of that bit
                idx = (lsb.bit_length() - 1)
                next_dist |= adj[idx]
                current_dist ^= lsb
            
            # Remove already visited nodes
            next_dist &= ~visited_mask
            dist[k] = next_dist
            visited_mask |= next_dist
            
            # If we reached v at any distance < 6, that's a shorter path, 
            # but we specifically care about path length 5 for C6.
            # Actually, if there is a path of length < 5, adding (u,v) creates a C3, C4, or C5.
            # The task says "containing no six-cycle". It does NOT forbid C3, C4, C5.
            # So we only care if there is a path of EXACTLY length 5?
            # No. If there is a path of length 3, adding (u,v) creates a C4. That is allowed.
            # If there is a path of length 5, adding (u,v) creates a C6. That is FORBIDDEN.
            # So we just need to check if v is in dist[5].
            
            if k == 5:
                if dist[5] & (1 << v):
                    return True
        
        return False

    # Main search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        adj = [0] * n
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u,v) creates a C6
            # Note: We must check in the context of the current graph 'adj'
            if not creates_c6(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Count edges
        edge_count = sum(bin(adj[i]).count("1") for i in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]
            
        # If we reached a high number, we might break early if time is tight, 
        # but usually more attempts help.
        # 105 is the target. 
        if best_edges >= 105:
             # We can continue to try for 106, 107...
             pass

    # Reconstruct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
