import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Build Grötzsch graph (11 vertices, 0..10)
    # Cycle 0-1-2-3-4-0
    # Copies 5-9
    # Top 10
    grotzsch_edges = []
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        grotzsch_edges.append((u, w))
        grotzsch_edges.append((u, 5 + w))
        grotzsch_edges.append((w, 5 + u))
    for u in range(5):
        grotzsch_edges.append((5 + u, 10))
        
    # Remove duplicates just in case
    grotzsch_edges = list(set(grotzsch_edges))
    
    n_vertices = 16
    # Initialize adjacency matrix
    adj = [[False] * n_vertices for _ in range(n_vertices)]
    for u, v in grotzsch_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # 2. Greedily add vertices 11..15
    # We want to maximize edges.
    # For each new vertex, connect to as many existing vertices as possible.
    # To avoid triangles, if we connect v to u, v cannot connect to any neighbor of u.
    # Greedy strategy:
    # Order of adding: 11, 12, 13, 14, 15.
    # For each new vertex v:
    #   available = set of all existing vertices (0..v-1)
    #   candidates = []
    #   for u in available:
    #       if not adj[v][u]:
    #           # Check if adding (v,u) creates a triangle
    #           # Triangle (v, u, w) exists if adj[u][w] and adj[v][w]
    #           # So u must not be adjacent to any current neighbor of v.
    #           # Actually, simpler: 
    #           # If we add (v,u), then for all w in N(v), w must not be in N(u).
    #           # And for all w in N(u), w must not be in N(v).
    #           # Since we are adding (v,u), N(v) is current neighbors of v.
    #           # So check: if any w in N(v) is in N(u), then (v,u) creates triangle.
    #           is_safe = True
    #           for w in range(n_vertices):
    #               if adj[v][w] and adj[u][w]:
    #                   is_safe = False
    #                   break
    #           if is_safe:
    #               candidates.append(u)
    #   # To maximize edges, we might want to choose a subset of candidates that are mutually non-adjacent?
    #   # No, if we connect v to u1 and u2, u1 and u2 must not be adjacent (otherwise triangle v-u1-u2).
    #   # So we need to find a maximal independent set in the subgraph induced by candidates?
    #   # Or just greedily pick one, then re-evaluate?
    #   # Let's just pick the candidate with highest degree or random, add it, and repeat?
    #   # Actually, we can add multiple edges at once if they form an independent set in the graph of candidates.
    #   # Let's just do a simple greedy:
    #   # While there is a safe candidate, pick one, add edge, update N(v).
    
    for v in range(11, 16):
        while True:
            candidates = []
            for u in range(v):
                if not adj[v][u]:
                    # Check safety
                    safe = True
                    for w in range(n_vertices):
                        if adj[v][w] and adj[u][w]:
                            safe = False
                            break
                    if safe:
                        candidates.append(u)
            
            if not candidates:
                break
            
            # Pick a candidate. Random or highest degree?
            # Random helps avoid local minima in the greedy phase.
            u = random.choice(candidates)
            adj[v][u] = True
            adj[u][v] = True
            
    # Current state is a triangle-free graph containing Grötzsch.
    # It is 4-chromatic because it contains Grötzsch.
    
    # 3. Local Search
    # Goal: Maximize edges.
    # Move 1: Add edge.
    # Move 2: Swap (remove 1, add 1 or more).
    
    best_edges = sum(1 for i in range(n_vertices) for j in range(i+1, n_vertices) if adj[i][j])
    
    # Helper to check if edge (u,v) can be added
    def can_add(u, v):
        if adj[u][v]:
            return False
        for w in range(n_vertices):
            if adj[u][w] and adj[v][w]:
                return False
        return True

    # Helper to get all addable edges
    def get_addable_edges():
        addable = []
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if can_add(u, v):
                    addable.append((u, v))
        return addable

    # Main loop
    while time.time() < deadline:
        addable = get_addable_edges()
        
        if addable:
            # Add a random edge
            u, v = random.choice(addable)
            adj[u][v] = True
            adj[v][u] = True
            best_edges += 1
        else:
            # No edges can be added. Try a swap.
            # Pick a random existing edge (u,v)
            all_edges = [(i, j) for i in range(n_vertices) for j in range(i+1, n_vertices) if adj[i][j]]
            if not all_edges:
                break
            
            u, v = random.choice(all_edges)
            
            # Remove it
            adj[u][v] = False
            adj[v][u] = False
            
            # Find new addable edges
            new_addable = get_addable_edges()
            
            # We want to add at least 1 edge. 
            # If we add 1, edge count stays same. Structure changes.
            # If we add 2, edge count increases.
            
            if new_addable:
                # Pick a random new edge to add
                new_u, new_v = random.choice(new_addable)
                adj[new_u][new_v] = True
                adj[new_v][new_u] = True
                # Edge count: -1 (removed u,v) + 1 (added new_u,new_v) = 0 change
                # But structure is different.
            else:
                # Revert
                adj[u][v] = True
                adj[v][u] = True
    
    # Construct final edge list
    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
