"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 135.0
    n = 40
    target_edges = 401
    
    # Start with K20,20 + (0,1)
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    edges.add((0, 1))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute book for each edge
    edge_book = {}
    for u, v in edges:
        key = (min(u,v), max(u,v))
        edge_book[key] = len(adj[u] & adj[v])
    
    current_max = max(edge_book.values()) if edge_book else 0
    best_max = current_max
    best_edges = edges.copy()
    
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i,j))
    
    random.seed(123)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        rem = random.choice(list(edges))
        ru, rv = min(rem[0],rem[1]), max(rem[0],rem[1])
        
        # Pick a random non-edge to add
        non_edges = all_edges - edges
        if not non_edges:
            break
        add = random.choice(list(non_edges))
        au, av = min(add[0],add[1]), max(add[0],add[1])
        
        # Affected vertices
        affected = {ru, rv, au, av}
        
        # Tentatively apply
        new_edges = edges.copy()
        new_edges.discard(rem)
        new_edges.add(add)
        
        if len(new_edges) != target_edges:
            continue
        
        # Build new adjacency
        new_adj = [set() for _ in range(n)]
        for u, v in new_edges:
            new_adj[u].add(v)
            new_adj[v].add(u)
        
        # Only recompute books for edges incident to affected vertices
        new_edge_book = {}
        for u, v in new_edges:
            key = (min(u,v), max(u,v))
            if u in affected or v in affected:
                new_edge_book[key] = len(new_adj[u] & new_adj[v])
            else:
                # Use old value
                if key in edge_book:
                    new_edge_book[key] = edge_book[key]
                else:
                    new_edge_book[key] = len(new_adj[u] & new_adj[v])
        
        new_max = max(new_edge_book.values()) if new_edge_book else 0
        
        if new_max < current_max:
            edges = new_edges
            adj = new_adj
            edge_book = new_edge_book
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_edges = edges.copy()
        elif new_max == current_max and random.random() < 0.4:
            edges = new_edges
            adj = new_adj
            edge_book = new_edge_book
    
    result = []
    for u, v in best_edges:
        if u > v:
            u, v = v, u
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
