import random
import time
from itertools import combinations

def construct():
    n = 16
    all_edges = list(combinations(range(n), 2))
    best_edges = []
    best_count = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Helper: check if adding edge (u,v) creates a triangle
    # We maintain adjacency lists or bitmasks for speed
    
    # We will run multiple random maximal triangle-free constructions
    # and check if they are 4-chromatic.
    
    def is_not_3_colorable(adj):
        # Backtracking 3-colorability check
        # adj is list of lists of neighbors
        colors = [-1] * n
        def backtrack(v):
            if v == n:
                return True
            # Try colors 0, 1, 2
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            for c in range(3):
                if c not in used:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        return not backtrack(0)

    def generate_maximal_tf():
        random.shuffle(all_edges)
        adj = [[] for _ in range(n)]
        adj_set = [set() for _ in range(n)]
        edges = []
        for u, v in all_edges:
            # Check if adding (u,v) creates a triangle
            # Common neighbors of u and v
            if adj_set[u].isdisjoint(adj_set[v]):
                # Add edge
                adj[u].append(v)
                adj[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append((u, v))
        return edges, adj

    # Try many random maximal triangle-free graphs
    # We can also try to bias the generation towards denser graphs?
    # Random order is fine.
    
    num_trials = 20000
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        edges, adj = generate_maximal_tf()
        # Check if 4-chromatic (not 3-colorable)
        if is_not_3_colorable(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
                
        # Optional: local search to add more edges?
        # Since it's maximal triangle-free, we can't add edges without creating triangles.
        # So this is the max for this specific "random greedy" construction.
        # Different random orders yield different maximal TF graphs.
        
    # If we didn't find any (unlikely), return seed
    if not best_edges:
        # Grötzsch graph
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    return [list(e) for e in best_edges]
