import random
import time

def construct():
    # Grötzsch graph edges
    # Vertices 0..4: Cycle
    # Vertices 5..9: Copies
    # Vertex 10: Top
    edges = set()
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.add((i, (i + 1) % 5))
        edges.add(((i + 1) % 5, i))
    # Connect u to 5+w and w to 5+u
    for i in range(5):
        u = i
        w = (i + 1) % 5
        edges.add((u, 5 + w))
        edges.add((5 + w, u))
        edges.add((w, 5 + u))
        edges.add((5 + u, w))
    # Connect 5+u to 10
    for u in range(5):
        edges.add((5 + u, 10))
        edges.add((10, 5 + u))
        
    # Convert to list for easier manipulation if needed, but set is good for lookup
    # We need adjacency lists for triangle check
    n = 16
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_edges = set(edges)
    best_count = len(edges)
    
    # Greedy addition with randomization
    # Candidate edges are all non-edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_found = best_count
    
    for _ in range(1000): # Multiple restarts
        if time.time() > deadline:
            break
            
        current_edges = set(best_edges)
        current_adj = [set() for _ in range(n)]
        for u, v in current_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        # Shuffle candidate edges
        candidates = [e for e in all_pairs if e not in current_edges and (e[1], e[0]) not in current_edges]
        random.shuffle(candidates)
        
        # Greedy add
        for u, v in candidates:
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            # Check triangle free: N(u) and N(v) disjoint
            if current_adj[u].isdisjoint(current_adj[v]):
                current_edges.add((u, v))
                current_edges.add((v, u))
                current_adj[u].add(v)
                current_adj[v].add(u)
                
        if len(current_edges) > best_found:
            best_found = len(current_edges)
            best_edges = set(current_edges)
            
    # Return list of edges [u, v]
    # Ensure unique undirected edges
    result = []
    for u, v in best_edges:
        if u < v:
            result.append([u, v])
            
    return result
