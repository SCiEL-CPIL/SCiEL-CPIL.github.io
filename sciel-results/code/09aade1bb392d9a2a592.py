import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges) # 120
    
    # Fast Q3 check using bitmasks
    # adj[i] is a bitmask of neighbors of i
    def creates_q3(u, v, adj):
        # If u and v are already connected, this is not an "addition" check, but we assume they are not.
        # We check if adding (u,v) creates a Q3.
        # A Q3 containing (u,v) must have u and v connected.
        # In Q3, every edge is in 4 faces (squares).
        # So there must be 4 common neighbors? No.
        # Let's use the structure:
        # Q3 vertices: {0,1}^3.
        # Edge (u,v).
        # The other 6 vertices must form specific connections.
        
        # Optimization from inspiration:
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Iterate a in N(u)
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a
            
            # Iterate b in N(u) \ {a}
            mask_b = mask_u & ~lsb_a
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                # c in N(v) \cap N(a)
                # d in N(v) \cap N(b)
                # In a cube, if (u,v) is an edge, and a,b are neighbors of u (other than v),
                # and c,d are neighbors of v...
                # Actually, the cube structure around edge (u,v):
                # u is connected to a, b (and v).
                # v is connected to c, d (and u).
                # a is connected to c? b is connected to d?
                # And a-c, b-d are edges?
                # And there is an edge between the "other" neighbors?
                
                # Let's use the specific check from the inspiration which worked well.
                mask_c = mask_v & adj[a]
                mask_d = mask_v & adj[b]
                
                if not mask_c or not mask_d:
                    continue
                
                # We need to find c in mask_c and d in mask_d such that:
                # 1. c != d
                # 2. (a, c) is edge (implied by mask_c)
                # 3. (b, d) is edge (implied by mask_d)
                # 4. There exists e, f such that (a,e), (b,e), (c,f), (d,f), (e,f) are edges?
                # This is getting complex. The inspiration's check was:
                # "Check if there is e in mask_e and f in mask_f such that e!=f and (e,f) edge."
                # where mask_e = N(a) \cap N(b) ...
                
                # Let's trust the inspiration's logic which scored 1.53.
                # mask_e = (adj[a] & adj[b]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                # This logic assumes a specific embedding of the cube.
                
                # To be safe and fast, I will implement a direct check for all 12 edges of a potential Q3
                # given the edge (u,v) and neighbors.
                
                # However, the inspiration's `creates_q3` is proven. I will reuse its core logic.
                
                mask_c_iter = mask_c
                while mask_c_iter:
                    lsb_c = mask_c_iter & -mask_c_iter
                    c = lsb_c.bit_length() - 1
                    mask_c_iter ^= lsb_c
                    
                    if c == a or c == b:
                        continue
                        
                    mask_d_iter = mask_d & ~lsb_c
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        if d == a or d == b or d == c:
                            continue
                        
                        # Now we have u, v, a, b, c, d.
                        # We need to check if the remaining 2 vertices e, f complete the cube.
                        # In Q3, the vertices are bipartite.
                        # u, a, b are in one part? No.
                        # u-v is an edge.
                        # u is connected to a, b.
                        # v is connected to c, d.
                        # a is connected to c? (mask_c check)
                        # b is connected to d? (mask_d check)
                        # We need e such that e is connected to a and b.
                        # We need f such that f is connected to c and d.
                        # And e is connected to f.
                        
                        mask_e = (adj[a] & adj[b]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_e == 0:
                            continue
                            
                        mask_f = (adj[c] & adj[d]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_f == 0:
                            continue
                        
                        # Check if there is e in mask_e and f in mask_f such that (e,f) is an edge.
                        if mask_e.bit_count() < mask_f.bit_count():
                            mask_e_iter = mask_e
                            while mask_e_iter:
                                lsb_e = mask_e_iter & -mask_e_iter
                                e = lsb_e.bit_length() - 1
                                mask_e_iter ^= lsb_e
                                if mask_f & adj[e] & ~lsb_e:
                                    return True
                        else:
                            mask_f_iter = mask_f
                            while mask_f_iter:
                                lsb_f = mask_f_iter & -mask_f_iter
                                f = lsb_f.bit_length() - 1
                                mask_f_iter ^= lsb_f
                                if mask_e & adj[f] & ~lsb_f:
                                    return True
        return False

    def greedy_construct(edge_order):
        adj = [0] * n
        edges = []
        
        for u, v in edge_order:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges

    def local_search(adj, edges, time_limit):
        # edges is a list of (u,v)
        # adj is bitmask array
        best_count = len(edges)
        best_adj = adj[:]
        best_edges = edges[:]
        
        # Precompute missing edges
        def get_missing_edges(adj):
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        missing.append((u, v))
            return missing
        
        # Simulated Annealing / Tabu-like local search
        current_adj = adj[:]
        current_edges = set(edges)
        current_count = len(current_edges)
        
        # We want to maximize count.
        # Move: Try to add an edge.
        # If stuck, try to remove one and add another.
        
        while time.time() < time_limit:
            missing = get_missing_edges(current_adj)
            if not missing:
                break
                
            random.shuffle(missing)
            
            improved = False
            for u, v in missing:
                if time.time() >= time_limit:
                    break
                if not creates_q3(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                        best_edges = list(current_edges)
                        
            if not improved:
                # Local optimum reached. Try a "kick"
                # Remove a random edge to try to add a different one
                if current_edges:
                    u, v = random.choice(list(current_edges))
                    # Only remove if we can potentially add more?
                    # Just remove and see if we can add back or add others.
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges.remove((u, v))
                    current_count -= 1
                    
                    # Try to add edges again
                    missing = get_missing_edges(current_adj)
                    random.shuffle(missing)
                    for u2, v2 in missing:
                        if not creates_q3(u2, v2, current_adj):
                            current_adj[u2] |= (1 << v2)
                            current_adj[v2] |= (1 << u2)
                            current_edges.add((u2, v2))
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_adj = current_adj[:]
                                best_edges = list(current_edges)
                                
        return best_adj, best_edges

    # Geometric Embedding for Initial Order
    # Place vertices on a sphere
    random.seed(42)
    points = []
    for _ in range(n):
        z = random.uniform(-1, 1)
        phi = random.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - z*z)
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        points.append((x, y, z))
        
    # Compute distances
    dists = {}
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dz = points[i][2] - points[j][2]
            dists[(i,j)] = math.sqrt(dx*dx + dy*dy + dz*dz)
            
    # Sort edges by distance (ascending)
    sorted_edges = sorted(all_edges, key=lambda e: dists[e])
    
    # Run Greedy with Geometric Order
    adj, edges = greedy_construct(sorted_edges)
    
    # Run Local Search
    time_left = deadline - time.time()
    if time_left > 5.0:
        adj, edges = local_search(adj, edges, deadline - 5.0)
        
    # Convert to list of lists
    result = [[u, v] for u, v in edges]
    return result
