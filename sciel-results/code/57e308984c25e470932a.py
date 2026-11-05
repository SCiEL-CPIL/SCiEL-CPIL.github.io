import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit (evaluator kills at 150s, but we want to be safe)
    
    n = 40
    m = 401
    
    # Structural Initialization:
    # Start with K_{20,20} (vertices 0-19 vs 20-39). 400 edges.
    # Add one edge within the first partition, e.g., (0, 1).
    # This creates triangles. The book size of (0,1) is 20.
    # We will use local search to improve this.
    
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Add bipartite edges
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    # Add the extra edge
    extra_edge = (0, 1)
    edges.add(extra_edge)
    adj[0].add(1)
    adj[1].add(0)
    
    def get_max_book():
        max_b = 0
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
    
    current_max = get_max_book()
    best_max = current_max
    best_edges_snapshot = list(edges)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Local Search
    # We try to reduce the max book size.
    # Strategy: Pick a random edge to remove and a random non-edge to add.
    # If the new max book is smaller, accept.
    # If equal, accept with small probability to allow exploration.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if steps % 1000 == 0:
            # Periodic check to avoid infinite loops if time is not advancing? No, time always advances.
            pass
        
        # Pick random edge to remove
        u1, v1 = random.choice(list(edges))
        
        # Pick random non-edge to add
        # Optimization: Instead of sampling until non-edge, we can maintain a list of non-edges?
        # Or just sample. N=40, M=401, total pairs=780. Density ~50%.
        # Sampling 10 times is very likely to find a non-edge.
        new_edge = None
        for _ in range(20):
            cand = random.choice(all_pairs)
            if cand not in edges:
                new_edge = cand
                break
        
        if new_edge is None:
            continue
            
        u2, v2 = new_edge
        
        # Perform Swap
        edges.remove((u1, v1))
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        
        edges.add(new_edge)
        adj[u2].add(v2)
        adj[v2].add(u2)
        
        # Calculate new max book
        new_max = get_max_book()
        
        # Acceptance
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges_snapshot = list(edges)
        elif new_max == current_max and random.random() < 0.1:
            # Accept equal moves to escape local minima
            pass
        else:
            # Revert
            edges.add((u1, v1))
            adj[u1].add(v1)
            adj[v1].add(u1)
            
            edges.remove(new_edge)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
    return best_edges_snapshot
