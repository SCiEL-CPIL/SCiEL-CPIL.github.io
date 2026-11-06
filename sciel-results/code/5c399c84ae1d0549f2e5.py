import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Base solution: 10 disjoint triangles
    def get_base():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_n = 30
    best_edges = get_base()['edges']
    
    # Try to improve via random search
    # We want n=39, alpha < 11
    # Greedy: Build graph vertex by vertex
    # Constraint: No C4.
    # Check: Alpha < 11.
    
    def check_c4(n, edges):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check for C4: Any two vertices have <= 1 common neighbor
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u].intersection(adj[v])
                    if len(common) > 1:
                        return False
        return True

    def max_independent_set_size(n, edges):
        # Greedy approximation for alpha
        # Or exact for small n? n=39 is too big for exact in Python without optimization.
        # Use greedy: sort by degree, pick min degree, remove neighbors.
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining = set(range(n))
        count = 0
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            count += 1
            remaining.remove(min_v)
            for u in adj[min_v]:
                if u in remaining:
                    remaining.remove(u)
        return count

    # Search loop
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        n = 39
        edges = []
        # Start with random sparse graph and add edges?
        # Or start with 10 triangles and add 9 vertices?
        
        # Strategy: Start with 10 triangles (30 verts). Add 9 new vertices.
        # Connect new vertices to existing to maintain C4-free and reduce alpha.
        
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
            
        # Add vertices 30..38
        for v in range(30, 39):
            # Try to connect v to some existing vertices
            # To reduce alpha, connect to vertices in different triangles
            # Randomly pick 2-3 neighbors from previous vertices
            candidates = list(range(0, v))
            random.shuffle(candidates)
            
            # Try adding edges one by one
            added = 0
            for u in candidates:
                if added >= 3: break
                # Check if adding (u,v) creates C4
                # C4 created if u and v have > 1 common neighbor
                # Common neighbors of u and v:
                # v is new, so neighbors of v are only those added so far.
                # Let N_v be current neighbors of v.
                # Common neighbors of u and v = N_v ∩ N_u.
                # We need len(N_v ∩ N_u) <= 1.
                
                # Get neighbors of u
                N_u = set()
                for e in edges:
                    if e[0] == u: N_u.add(e[1])
                    elif e[1] == u: N_u.add(e[0])
                
                # Get neighbors of v (so far)
                N_v = set()
                for e in edges:
                    if e[0] == v: N_v.add(e[1])
                    elif e[1] == v: N_v.add(e[0])
                
                common = N_u.intersection(N_v)
                if len(common) <= 1:
                    edges.append([u, v])
                    added += 1
        
        # Verify C4-free
        if check_c4(n, edges):
            # Check Alpha
            alpha = max_independent_set_size(n, edges)
            if alpha < 11:
                best_n = n
                best_edges = edges
                break # Found a valid 39-vertex graph
        
    return {'n': best_n, 'edges': best_edges}
