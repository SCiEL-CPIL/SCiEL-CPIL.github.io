import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # Partition sizes
    # A: size 9 (vertices 0..8)
    # B: size 31 (vertices 9..39)
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size)
    # 122
    
    # Deterministic greedy for B-B edges
    # We want to find a triangle-free graph with 122 edges and max degree 8.
    # Heuristic: Prioritize edges between vertices with lowest current degree.
    
    B_edges = []
    degree = {v: 0 for v in B}
    adj = {v: set() for v in B}
    
    # Generate all pairs
    pairs = list(itertools.combinations(B, 2))
    
    # To make it deterministic and potentially better, we can sort pairs
    # But sorting 465 pairs is cheap.
    # We don't sort by degree initially, but we re-evaluate or use a priority queue?
    # Simple approach: Iterate, but pick the "best" available edge each time?
    # That would be O(E^2) or similar. 122 edges, 31 vertices.
    # 122 * 465 checks is fine.
    
    # However, a simpler deterministic greedy is just iterating in a fixed order
    # that tends to balance degrees, e.g., round-robin or sorted by vertex index.
    # The random shuffle was the only non-determinism.
    # Let's try a sorted order by (u, v).
    
    # Actually, to maximize slack, we should add edges that connect low-degree vertices.
    # We can use a simple loop:
    # While we need edges:
    #   Find the pair (u,v) not yet considered/added, no common neighbor, 
    #   and min(degree[u], degree[v]) is minimal? Or sum?
    #   Add it.
    
    # This "Best-First" approach is more robust than random.
    
    candidates = set(pairs)
    
    while len(B_edges) < needed_in_B:
        if time.time() > deadline:
            break
            
        best_pair = None
        best_score = None
        
        # Iterate all remaining candidates
        # To speed up, we can just iterate and keep the best one.
        for u, v in candidates:
            if (u, v) in adj[u]: # Already added? No, adj is set of neighbors.
                continue
            
            # Check if already in B_edges?
            # We remove from candidates when added.
            
            # Check triangle
            if adj[u] & adj[v]:
                continue
                
            # Check degree
            # We aim for max degree 8.
            if degree[u] >= 8 or degree[v] >= 8:
                continue
                
            # Score: Prefer pairs where both degrees are low.
            # Sum of degrees is a good proxy for "slack".
            score = degree[u] + degree[v]
            
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (u, v)
                
        if best_pair is None:
            # No valid edge with max degree 8.
            # Relax to 9?
            # If we can't find 8, we try 9.
            # But if we are in the "try 8" phase, we should restart or relax.
            # Let's relax to 9 immediately if we get stuck.
            # We need to restart the search with limit 9 if we fail to fill 122 with 8.
            # For now, if we can't find ANY edge with limit 8, we break and try limit 9.
            break
            
        # Add the best pair
        u, v = best_pair
        B_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        candidates.remove((u, v))
        
    if len(B_edges) < needed_in_B:
        # Restart with limit 9
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        candidates = set(pairs)
        
        while len(B_edges) < needed_in_B:
            if time.time() > deadline:
                break
                
            best_pair = None
            best_score = None
            
            for u, v in candidates:
                if adj[u] & adj[v]:
                    continue
                    
                if degree[u] >= 9 or degree[v] >= 9:
                    continue
                    
                score = degree[u] + degree[v]
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (u, v)
                    
            if best_pair is None:
                break
                
            u, v = best_pair
            B_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            candidates.remove((u, v))
            
    for u, v in B_edges:
        edges.add((u, v))
        
    result = [[u, v] for u, v in edges]
    
    return result
