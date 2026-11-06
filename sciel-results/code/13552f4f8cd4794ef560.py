def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # Build the polarity graph of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, (y * inv_x) % 5, (z * inv_x) % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, (z * inv_y) % 5))
                else:
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_base = len(points)
    
    def bilinear(u, v):
        return (u[0] * v[2] + u[1] * v[1] + u[2] * v[0]) % 5
    
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)
    
    # Precompute compatible pairs: (a, b) such that N(a) & N(b) == 0
    compatible_pairs = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if adj_base[i] & adj_base[j] == 0:
                # Check common neighbors
                if (adj_base[i] & adj_base[j]) == 0: # This is always true if non-adjacent? No.
                     # Wait, if they are non-adjacent, they can still have common neighbors.
                     # Condition for C4: N(a) & N(b) == 0
                     pass
            
            # Correct condition: N(a) and N(b) must be disjoint.
            # Note: a is not in N(a) in simple graph adjacency mask?
            # adj_base[i] contains neighbors. i is not in adj_base[i].
            # So we just check if adj_base[i] & adj_base[j] == 0.
            if (adj_base[i] & adj_base[j]) == 0:
                compatible_pairs.append((i, j))
    
    # Shuffle compatible pairs for variety
    import random
    random.seed(42)
    random.shuffle(compatible_pairs)
    
    def is_c4_free(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def max_independent_set_size(n, adj):
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')

        def branch(candidates_mask, current_size):
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    best_graph = None
    best_n = n_base
    adj_best = adj_base[:]

    # Try adding 8 to 12 vertices
    for num_new in range(12, 7, -1):
        if time.time() > deadline:
            break
            
        n_total = n_base + num_new
        adj = [0] * n_total
        for i in range(n_base):
            adj[i] = adj_base[i]
        
        # New vertices are independent (no edges between them)
        # Connect each new vertex to a compatible pair from base
        
        # We need to select num_new pairs from compatible_pairs
        # To avoid conflicts, we can just pick distinct pairs?
        # If two new vertices connect to the same base vertex, is that a C4?
        # u1-a-u2-...? No, u1 and u2 are not connected.
        # u1-a-b-u1 is not a cycle if u1 only connects to a,b.
        # If u1 connects to a,b and u2 connects to a,c.
        # C4: u1-a-u2-c-...? No, u1-c is not an edge.
        # So sharing base vertices is fine as long as no C4 is formed.
        # C4 involving new vertices:
        # u1 - a - u2 - b - u1?
        # Requires u1-b edge. u1 connects to a,b? Yes.
        # u2-a edge? Yes.
        # So if u1 connects to {a,b} and u2 connects to {a,c}, and a,b,c are in base.
        # Is u1-a-u2-b-u1 a C4?
        # Edges: (u1,a), (a,u2), (u2,b), (b,u1).
        # Vertices: u1, a, u2, b.
        # This is a C4 if a != b and u1 != u2.
        # So we must ensure that no two new vertices share exactly one neighbor, or if they do, the other neighbors don't form a C4.
        # Actually, if u1 ~ a,b and u2 ~ a,c.
        # Common neighbors of u1 and u2: {a}.
        # Common neighbors of a and b: ?
        # The C4 is u1-a-u2-b-u1.
        # This requires u2-b edge? No, u2 is connected to a and c.
        # So (u2, b) is NOT an edge.
        # So u1-a-u2-b-u1 is NOT a cycle.
        # What is a C4?
        # u1 - x - u2 - y - u1.
        # x, y must be neighbors of u1 and u2 respectively?
        # x is neighbor of u1. y is neighbor of u2.
        # x must be neighbor of u2? No, x is neighbor of u1 and u2?
        # If x is a common neighbor of u1 and u2, then u1-x-u2-...
        # If u1 and u2 share a neighbor x, and u1 has another neighbor y, and u2 has another neighbor z.
        # If y and z are connected? No.
        # If y=z? Then u1-y-u2-x-u1?
        # If u1 ~ x,y and u2 ~ x,y.
        # Then u1-x-u2-y-u1 is a C4.
        # So we must not have two new vertices with the SAME pair of neighbors.
        # If we pick distinct pairs, we are safe?
        # If u1 ~ a,b and u2 ~ c,d.
        # If a=c, then they share a.
        # Do they form a C4?
        # u1-a-u2-...?
        # u2 is connected to c(=a) and d.
        # u1 is connected to a and b.
        # Common neighbors of u1 and u2: {a}.
        # Is there a C4?
        # u1-a-u2-d-...? d is not connected to u1.
        # u1-b-...? b is not connected to u2.
        # So no C4.
        
        # So we can just pick distinct compatible pairs.
        
        # Let's pick num_new distinct pairs from compatible_pairs
        if len(compatible_pairs) < num_new:
            continue
            
        selected_pairs = compatible_pairs[:num_new]
        
        for k in range(num_new):
            new_v = n_base + k
            a, b = selected_pairs[k]
            adj[new_v] |= (1 << a)
            adj[a] |= (1 << new_v)
            adj[new_v] |= (1 << b)
            adj[b] |= (1 << new_v)
            
        if not is_c4_free(n_total, adj):
            continue
            
        alpha = max_independent_set_size(n_total, adj)
        
        if alpha < 11:
            best_graph = adj
            best_n = n_total
            break # Found a valid larger graph
            
    if best_graph is None:
        # Fallback to base
        best_n = n_base
        best_graph = adj_base[:]
        
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_graph[i] & (1 << j):
                edge_list.append([i, j])
                
    return {'n': best_n, 'edges': edge_list}
