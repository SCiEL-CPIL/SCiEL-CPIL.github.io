import random
import time

def construct():
    n = 20
    # List of all possible edges
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Deadline guard
    start_time = time.time()
    max_time = 140.0 # Leave 10s margin
    
    # Number of restarts. We can do many because each run is fast.
    # Let's do 1000 restarts if time permits.
    attempts = 0
    
    while time.time() - start_time < max_time:
        random.shuffle(edges)
        
        # Adjacency matrix or sets for fast lookup
        # Using sets of neighbors
        adj = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in edges:
            # Check if adding (u, v) creates a C4
            # A C4 is u-v-w-x-u. 
            # We need w in N(v) and x in N(u) such that (w,x) is an edge.
            # Also w != x, w != u, x != v.
            creates_c4 = False
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[u]:
                    if x == v or x == w:
                        continue
                    if x in adj[w]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                current_edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
        
        attempts += 1
        # If we hit a very high number, we might stop early? 
        # ex(20, C4) is 86. If we find 86, we can stop.
        if best_count >= 86:
            break

    # Assign colors
    # Since there are no C4s, any coloring is valid.
    # We just need to use colors 0..19.
    colored_edges = []
    for i, (u, v) in enumerate(best_edges):
        c = i % 20
        colored_edges.append([u, v, c])
        
    return colored_edges
