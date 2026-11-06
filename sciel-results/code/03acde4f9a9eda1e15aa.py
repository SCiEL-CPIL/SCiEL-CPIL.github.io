import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Grötzsch graph edges (11 vertices 0-10)
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,1), (5,4),
        (6,0), (6,2),
        (7,1), (7,3),
        (8,2), (8,4),
        (9,0), (9,3),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    
    def init_adj(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def is_triangle_free(adj, u, v):
        # Check if adding (u,v) creates a triangle
        if (adj[u] >> v) & 1:
            return False # Already an edge
        if (adj[u] & adj[v]) != 0:
            return False
        return True

    def is_3_colorable(adj, n):
        # Backtracking 3-colorability check
        colors = [-1] * n
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            u = order[idx]
            for c in range(3):
                valid = True
                for v in range(n):
                    if (adj[u] >> v) & 1 and colors[v] == c:
                        valid = False
                        break
                if valid:
                    colors[u] = c
                    if backtrack(idx + 1):
                        return True
                    colors[u] = -1
            return False
            
        return backtrack(0)

    def greedy_random_complete(adj, n, rng):
        adj = adj[:]
        while True:
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if is_triangle_free(adj, u, v):
                        candidates.append((u, v))
            if not candidates:
                break
            u, v = rng.choice(candidates)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    best_adj = None
    best_edges = 0
    
    # Initial solution: Grötzsch + greedy fill
    adj_core = init_adj(grotzsch_edges, n)
    rng = random.Random(0)
    adj_full = greedy_random_complete(adj_core, n, rng)
    
    # Verify initial is 4-chromatic (it should be, but just in case)
    if not is_3_colorable(adj_full, n):
        best_adj = adj_full[:]
        best_edges = count_edges(adj_full)
    else:
        # Fallback if something went wrong with the core
        best_adj = adj_core[:]
        best_edges = count_edges(adj_core)

    num_trials = 5000
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        # Local search from current best
        # Pick a random edge to remove
        all_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (best_adj[u] >> v) & 1:
                    all_edges.append((u,v))
        
        if not all_edges:
            break
            
        u, v = rng.choice(all_edges)
        
        # Remove
        adj_temp = best_adj[:]
        adj_temp[u] &= ~(1 << v)
        adj_temp[v] &= ~(1 << u)
        
        # Complete
        adj_new = greedy_random_complete(adj_temp, n, rng)
        new_edges = count_edges(adj_new)
        
        # Check 3-colorability
        if new_edges > best_edges:
            if not is_3_colorable(adj_new, n):
                best_edges = new_edges
                best_adj = adj_new[:]
            # If it is 3-colorable, we ignore it (don't update best_adj)
            # This ensures best_adj is always 4-chromatic

    # Convert best_adj to edge list
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
                
    return edges_list
