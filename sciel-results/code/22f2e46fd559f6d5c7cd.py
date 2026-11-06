import random
import time

def construct():
    # Grötzsch graph construction
    # Vertices 0-4: Cycle C5
    # Vertices 5-9: Shadows of 0-4
    # Vertex 10: Root
    # Vertices 11-15: Isolated
    
    initial_edges = set()
    # Cycle
    for i in range(5):
        u, v = i, (i + 1) % 5
        initial_edges.add((min(u, v), max(u, v)))
    
    # Shadows
    for i in range(5):
        u = i
        v = (i + 1) % 5
        # u connected to shadow of v (5+v)
        a, b = u, 5 + v
        initial_edges.add((min(a, b), max(a, b)))
        # v connected to shadow of u (5+u)
        a, b = v, 5 + u
        initial_edges.add((min(a, b), max(a, b)))
        
    # Root connected to all shadows
    for i in range(5):
        a, b = 10, 5 + i
        initial_edges.add((min(a, b), max(a, b)))
        
    n = 16
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = set(initial_edges)
    best_count = len(best_edges)
    
    def creates_triangle(u, v, edge_set):
        # Check if there exists w such that (u,w) and (v,w) are in edge_set
        for w in range(n):
            if w == u or w == v:
                continue
            e1 = (min(u, w), max(u, w))
            e2 = (min(v, w), max(v, w))
            if e1 in edge_set and e2 in edge_set:
                return True
        return False

    def is_3_colorable(edge_set):
        # Backtracking 3-coloring
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
            
        colors = [-1] * n
        
        def backtrack(i):
            if i == n:
                return True
            for c in range(3):
                valid = True
                for j in range(i):
                    if adj[i][j] and colors[j] == c:
                        valid = False
                        break
                if valid:
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
            return False
            
        return backtrack(0)

    deadline = time.time() + 145
    
    # We can also try to remove edges and re-add? 
    # No, adding edges is monotonic for non-3-colorability.
    # So we just want to maximize edges subject to triangle-free.
    
    # Greedy fill with random restarts
    while time.time() < deadline:
        # Start from current best? Or from initial?
        # Starting from current best allows incremental improvement.
        # But greedy fill might get stuck in local optimum.
        # Let's try starting from initial for some iterations, and from best for others.
        
        if random.random() < 0.5:
            start_edges = set(initial_edges)
        else:
            start_edges = set(best_edges)
            
        # Get missing edges
        missing = [e for e in all_possible_edges if e not in start_edges]
        random.shuffle(missing)
        
        trial_edges = set(start_edges)
        
        for u, v in missing:
            if not creates_triangle(u, v, trial_edges):
                trial_edges.add((u, v))
                
        if len(trial_edges) > best_count:
            best_count = len(trial_edges)
            best_edges = trial_edges
            
    # Final check
    if is_3_colorable(best_edges):
        # Should not happen
        return [[u, v] for u, v in initial_edges]
        
    return [[u, v] for u, v in best_edges]
