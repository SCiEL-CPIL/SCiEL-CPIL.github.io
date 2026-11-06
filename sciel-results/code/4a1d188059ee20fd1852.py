import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    best_edges = 0
    best_graph = None
    
    # Try to find a C4-free graph with 42 edges
    # We use a greedy approach with random restarts
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Randomize edge order
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        
        # Adjacency matrix
        adj = [[False] * n for _ in range(n)]
        current_edges = []
        
        for u, v in edges:
            # Check if adding (u, v) creates a C4
            # A C4 is u-v-w-x-u. 
            # Existing edges: u-x, x-w, w-v.
            # So we need to check if there is a path of length 3 from u to v.
            # Path: u - a - b - v.
            # a is neighbor of u, b is neighbor of v, (a,b) is edge.
            
            creates_c4 = False
            for a in range(n):
                if adj[u][a]:
                    for b in range(n):
                        if adj[v][b] and adj[a][b]:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
            
            if not creates_c4:
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append((u, v))
                
        if len(current_edges) > best_edges:
            best_edges = len(current_edges)
            best_graph = current_edges
            if best_edges >= 42:
                break
                
    if best_graph is None:
        # Fallback to star
        return [[0, v, 0] for v in range(1, n)]
        
    # Color all edges with 0 (valid since no C4s)
    # But we have 20 colors available. We can use any.
    # The problem says "edge-coloured with 20 colours".
    # It doesn't say we must use all 20.
    # "every 4-cycle is rainbow". If no 4-cycles, condition holds.
    return [[u, v, 0] for u, v in best_graph]
