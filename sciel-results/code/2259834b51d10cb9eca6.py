import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Target configurations: (n, min_degree_required)
    # We want to maximize n.
    # n=46 -> deg 7
    # n=45 -> deg 6
    # n=44 -> deg 5
    # n=43 -> deg 4
    # n=42 -> deg 3
    # n=41 -> deg 2
    # n=40 -> deg 1
    # n=39 -> deg 0 (Empty graph)
    
    targets = [
        (46, 7),
        (45, 6),
        (44, 5),
        (43, 4),
        (42, 3),
        (41, 2),
        (40, 1),
        (39, 0)
    ]
    
    for n, k in targets:
        if time.time() > deadline:
            break
            
        result = search_c4_free(n, k, deadline)
        if result:
            return result
            
    # Fallback: Empty graph on 39 vertices
    return {'n': 39, 'edges': []}

def search_c4_free(n, min_deg, deadline):
    """
    Searches for a C4-free graph on n vertices with min degree >= min_deg.
    """
    # Try a few random restarts
    max_restarts = 100
    restarts = 0
    while restarts < max_restarts and time.time() < deadline:
        random.seed(restarts)
        
        # Initialize graph
        adj = [0] * n
        
        # Start with a sparse random graph
        # Probability tuned to be C4-free-ish
        # For small n, p=0.1 is safe
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove any C4s
        remove_c4s(adj, n)
        
        # Try to increase min degree
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find vertices with degree < min_deg
            low_deg_verts = [i for i in range(n) if bin(adj[i]).count('1') < min_deg]
            
            if not low_deg_verts:
                break
                
            # Try to add an edge for the first low degree vertex
            u = low_deg_verts[0]
            # Candidates: vertices v not adjacent to u
            candidates = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                # Check if adding (u,v) creates C4
                # Condition: |N(u) & N(v)| <= 1
                if (adj[u] & adj[v]).bit_count() <= 1:
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    added = True
                    break
                    
            if not added:
                # Stuck
                break
        
        # Check if success
        if all(bin(adj[i]).count('1') >= min_deg for i in range(n)):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        restarts += 1
        
    return None

def remove_c4s(adj, n):
    """
    Removes edges that are part of any C4.
    """
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Check if edge (i,j) is part of a C4
                    # i and j have >= 2 common neighbors
                    common = adj[i] & adj[j]
                    if common.bit_count() >= 2:
                        # Remove edge (i,j)
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        changed = True
                        break
            if changed:
                break

