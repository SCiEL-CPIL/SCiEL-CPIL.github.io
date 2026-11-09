import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def is_c4_free(n, edges_set):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check common neighbors
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u].intersection(adj[v])
                    if len(common) > 1:
                        return False
        return True

    def get_edges_list(edges_set):
        return [[u, v] for u, v in edges_set]

    def try_construct(n):
        # Target min degree
        target_deg = n - 39
        
        # Initialize with a random graph with avg degree slightly higher
        # We want to end up with min degree >= target_deg
        # Let's start with a random 8-regular-ish graph for n=46
        # Or just start empty and add edges?
        # Adding edges is safer for C4-free if we check carefully.
        
        adj = [set() for _ in range(n)]
        edges_set = set()
        
        # Helper to check if adding (u,v) creates C4
        # It creates C4 if |N(u) ∩ N(v)| >= 1? 
        # No, if u,v not connected, and they share x, y, then C4.
        # If u,v connected, and they share x, y, then C4.
        # So in all cases, we need |N(u) ∩ N(v)| <= 1 BEFORE adding?
        # If we add (u,v), the new common neighbors are N(u) ∩ N(v) \ {u,v} (which is empty since u not in N(u)).
        # Wait, N(u) does not contain u.
        # So if we add (u,v), the common neighbors of u and v in the NEW graph are N_old(u) ∩ N_old(v).
        # If len(N_old(u) ∩ N_old(v)) >= 2, then adding (u,v) creates a C4?
        # Let's trace: u-x-v-y-u.
        # Edges: (u,x), (x,v), (v,y), (y,u).
        # If we add (u,v), we form a triangle u-v-x-u if x is common neighbor.
        # If there are TWO common neighbors x, y, we have u-v-x-u AND u-v-y-u.
        # This doesn't form a C4 directly from the edge (u,v).
        # The C4 is u-x-v-y-u. This C4 exists REGARDLESS of whether (u,v) is present, 
        # PROVIDED that (u,x), (x,v), (v,y), (y,u) are present.
        # So if u and v have 2 common neighbors, the C4 already exists in the graph (without edge uv).
        # Therefore, a C4-free graph MUST have |N(u) ∩ N(v)| <= 1 for ALL u,v.
        # So we can just maintain this invariant.
        
        def can_add(u, v):
            if v in adj[u]:
                return False
            # Check if u and v have > 1 common neighbor
            common = adj[u].intersection(adj[v])
            return len(common) <= 1

        # Greedy construction
        # Try to add edges to increase degrees
        # Randomize vertex order
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # We will try to boost degrees to at least target_deg
        # Start with empty graph
        for _ in range(10000): # Iterations
            if time.time() > deadline:
                break
            
            # Find vertex with lowest degree
            min_deg = min(len(adj[v]) for v in vertices)
            if min_deg >= target_deg:
                # Check if C4 free (should be by invariant)
                return adj, edges_set
            
            # Pick a vertex with low degree
            low_v = random.choice([v for v in vertices if len(adj[v]) < target_deg])
            
            # Try to add an edge to low_v
            candidates = [v for v in vertices if v != low_v and v not in adj[low_v] and can_add(low_v, v)]
            if candidates:
                # Choose candidate that maximizes degree of neighbor? Or random?
                # Choose random to diversify
                new_v = random.choice(candidates)
                adj[low_v].add(new_v)
                adj[new_v].add(low_v)
                edges_set.add((min(low_v, new_v), max(low_v, new_v)))
            else:
                # Cannot add edge to low_v without violating C4-free
                # Maybe we need to remove some edges elsewhere to free up space?
                # Or just accept this graph if it's close?
                # For now, shuffle and try again
                pass

        # If we finish loop, return current best
        return adj, edges_set

    # Try n=46
    for n in [46, 45, 44]:
        target_deg = n - 39
        if target_deg < 1:
            continue
            
        adj, edges_set = try_construct(n)
        
        # Verify
        min_deg = min(len(adj[v]) for v in range(n))
        if min_deg >= target_deg:
            # Final C4 check
            if is_c4_free(n, edges_set):
                return {'n': n, 'edges': get_edges_list(edges_set)}
        
        if time.time() > deadline:
            break
            
    # Fallback: n=39 empty graph
    return {'n': 39, 'edges': []}
