import random
import time

def construct():
    # Grötzsch graph edges
    # Vertices 0-4: cycle
    # Vertices 5-9: connected to 0-4
    # Vertex 10: connected to 5-9
    edges = set()
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.add((i, (i + 1) % 5))
    
    # 5-9 connected to 0-4
    # 5 connected to 0, 4? No, standard Groetzsch construction:
    # Mycielski of C5.
    # Original C5: 0-1-2-3-4-0
    # New vertices 5-9.
    # 5 connected to 0, 4?
    # Let's stick to the seed's construction to be sure it's Groetzsch.
    # Seed:
    # cycle = [(i, (i + 1) % 5) for i in range(5)]
    # for u, w in cycle: edges += [[u, 5 + w], [w, 5 + u]]
    # This connects u to 5+w and w to 5+u.
    # e.g. (0,1) -> (0, 5+1=6), (1, 5+0=5).
    # So 0-6, 1-5.
    # (1,2) -> (1,7), (2,6).
    # (2,3) -> (2,8), (3,7).
    # (3,4) -> (3,9), (4,8).
    # (4,0) -> (4,5), (0,9).
    # And 10 connected to 5,6,7,8,9.
    
    # Let's rebuild the seed's graph exactly.
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        edges.add((u, w))
        edges.add((u, 5 + w))
        edges.add((w, 5 + u))
    for u in range(5):
        edges.add((5 + u, 10))
        
    # Ensure edges are (min, max) for set
    edges = {tuple(sorted(e)) for e in edges}
    
    n = 16
    all_possible_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                all_possible_edges.append((i, j))
                
    best_edges = set(edges)
    best_count = len(edges)
    
    start_time = time.time()
    deadline = 110  # 110 seconds to be safe
    
    # Random restarts
    while time.time() < deadline:
        current_edges = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Shuffle candidate edges
        candidates = list(all_possible_edges)
        random.shuffle(candidates)
        
        # Try to add edges
        for u, v in candidates:
            # Check if adding (u,v) creates a triangle
            # A triangle is formed if there exists w such that (u,w) and (v,w) are in current_edges
            # i.e., common neighbors of u and v in current_edges
            if not (adj[u] & adj[v]):
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    # Convert back to list of lists
    return [list(e) for e in best_edges]
