import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    
    # Helper to get edge count
    def get_edge_count(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # Helper to find a path of length 5 between u and v
    # Returns list of vertices [u, a, b, c, d, v] or None
    def find_path_len5(adj, u, v):
        # BFS from u up to depth 2
        # BFS from v up to depth 3
        # Actually, we can just do BFS from u to depth 5 and see if we hit v.
        # But we need the path.
        
        # Since n is small, we can do a simple DFS/BFS for path of length exactly 5.
        # This might be slow if called frequently.
        
        # Let's stick to the bitset check for speed, and only find the path if we need to kick.
        # For now, just return True/False.
        n2_u = 0
        for nb in range(n):
            if (adj[u] >> nb) & 1:
                n2_u |= adj[nb]
        n2_u &= ~(1 << u)
        
        n1_v = adj[v]
        n2_v = 0
        for nb in range(n):
            if (n1_v >> nb) & 1:
                n2_v |= adj[nb]
        n2_v &= ~(1 << v)
        
        n3_v = 0
        for nb in range(n):
            if (n2_v >> nb) & 1:
                n3_v |= adj[nb]
        n3_v &= ~(1 << v)
        
        return bool(n2_u & n3_v)

    # Initialize with a random greedy graph
    adj = [0] * n
    possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(possible_edges)
    
    for u, v in possible_edges:
        if not find_path_len5(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    best_adj = adj[:]
    best_count = get_edge_count(adj)
    
    # Local Search with Kicks
    while time.time() < deadline:
        # Try to add a random edge
        u, v = random.choice(possible_edges)
        if (adj[u] >> v) & 1:
            continue # Already exists
            
        if not find_path_len5(adj, u, v):
            # Safe to add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_count = get_edge_count(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        else:
            # Conflict. Kick: remove a random edge from the graph
            # To be smarter, we could remove an edge from the conflicting path,
            # but finding the path is expensive.
            # Random removal is a valid kick.
            existing_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        existing_edges.append((i, j))
                        
            if existing_edges:
                ru, rv = random.choice(existing_edges)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                # Now try to add (u, v) again? Or just continue loop?
                # Let's try to add (u, v) immediately
                if not find_path_len5(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count = get_edge_count(adj)
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]

    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
                
    return result_edges
