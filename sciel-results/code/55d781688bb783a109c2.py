import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit to be safe

    # Helper: Check if graph is C4-free
    # adj: list of int bitmasks
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    return False
        return True

    # Helper: Find a C4 and return one edge to remove
    def find_c4_edge(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    # Find two bits in common
                    bits = []
                    mask = common
                    while mask:
                        lsb = mask & -mask
                        bits.append(lsb.bit_length() - 1)
                        mask ^= lsb
                        if len(bits) == 2:
                            break
                    # u, v, bits[0], bits[1] form a C4
                    # Remove edge (u, bits[0])
                    return (u, bits[0])
        return None

    # Helper: Check if alpha < 11 (i.e., no independent set of size 11)
    # We check for clique of size 11 in complement
    def has_independent_set_11(adj, n):
        full_mask = (1 << n) - 1
        # non_neighbors[i] contains vertices NOT connected to i, and i itself is removed
        non_neighbors = [0] * n
        for i in range(n):
            # adj[i] has bits for neighbors
            # full_mask ^ adj[i] ^ (1<<i) gives non-neighbors excluding i
            non_neighbors[i] = full_mask ^ adj[i] ^ (1 << i)
            
        # Find clique of size 11 in graph defined by non_neighbors
        # candidates is a list of vertices
        def backtrack(candidates, size):
            if size == 11:
                return True
            if len(candidates) < 11 - size:
                return False
            
            # Heuristic: Pick vertex with max degree in the subgraph induced by candidates
            # Degree in complement subgraph = number of neighbors in candidates
            best_v = -1
            best_deg = -1
            for v in candidates:
                # Count neighbors of v in candidates
                mask = non_neighbors[v]
                deg = 0
                for u in candidates:
                    if (mask >> u) & 1:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            rest = [u for u in candidates if u != v]
            
            # Option 1: Include v
            new_candidates = [u for u in rest if (non_neighbors[v] >> u) & 1]
            if backtrack(new_candidates, size + 1):
                return True
            
            # Option 2: Exclude v
            if backtrack(rest, size):
                return False
            
            return False
            
        return backtrack(list(range(n)), 0)

    # Helper: Try to add an edge (u, v) to adj if it doesn't create a C4
    # Returns True if added, False otherwise
    def try_add_edge(adj, n, u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check if adding (u, v) creates a C4
        # A C4 is formed if there exists w such that w is neighbor of u and v
        # AND there exists x such that x is neighbor of u and v? No.
        # Adding (u,v) creates a C4 if there is a path u-w-x-v of length 3 that becomes u-w-x-v-u?
        # No, C4 is a cycle of length 4.
        # If we add (u,v), we form a C4 if there exists w such that w is a common neighbor of u and v.
        # Wait. If w is a common neighbor of u and v, then u-w-v is a path of length 2.
        # Adding (u,v) closes the triangle u-w-v-u. That's a C3, not C4.
        # C4 involves 4 distinct vertices.
        # u-v-w-x-u.
        # If we add (u,v), we might complete a C4 u-v-w-x-u where u-x, x-w, w-v are existing edges.
        # So, we need to check if there exists w, x such that:
        # 1. w is neighbor of v
        # 2. x is neighbor of u
        # 3. w is neighbor of x
        # 4. w != x, w != u, x != v
        # Basically, is there a common neighbor of u and v in the "line graph" sense?
        # No, simpler:
        # For each neighbor w of v, and each neighbor x of u:
        #   if w == x: continue (this would be a triangle, which is allowed)
        #   if (adj[w] >> x) & 1: 
        #       Then u-v-w-x-u is a C4.
        
        neighbors_v = []
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            neighbors_v.append(lsb.bit_length() - 1)
            mask_v ^= lsb
            
        neighbors_u = []
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            neighbors_u.append(lsb.bit_length() - 1)
            mask_u ^= lsb
            
        for w in neighbors_v:
            if w == u: continue
            for x in neighbors_u:
                if x == v: continue
                if w == x: continue
                if (adj[w] >> x) & 1:
                    return False
        return True

    # Main search loop
    best_n = 0
    best_edges = []
    
    # Try n from 39 down to 31
    for n in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts for this n
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Initialize empty graph
            adj = [0] * n
            
            # Phase 1: Build a random C4-free graph
            # We can add edges randomly, but ensure no C4.
            # To make it "dense" enough to reduce alpha, we should try to add many edges.
            # Strategy: Iterate through all pairs, shuffle them, and try to add.
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if try_add_edge(adj, n, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            # Phase 2: Check alpha
            if not has_independent_set_11(adj, n):
                # Valid graph found
                edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        mask ^= lsb
                best_n = n
                best_edges = edges
                found = True
                break
            
            # If alpha >= 11, we could try to fix it by adding more edges, 
            # but the graph is already maximal C4-free (in terms of the random order).
            # We might need a different random seed/order.
            
        if found:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback: n=31 Polarity Graph of PG(2,5)
    # This is a known C4-free graph with alpha=8
    def get_polarity_graph_q5():
        n = 31
        adj = [0] * n
        # Points of PG(2,5) can be represented as homogeneous coordinates [x:y:z] over GF(5)
        # There are 31 points.
        # Polarity: x*x' + y*y' + z*z' = 0 (mod 5) is a standard symplectic form?
        # For PG(2,q), a polarity is given by a symmetric matrix A.
        # A = [[1, 0, 0], [0, 1, 0], [0, 0, -1]] works for q=5?
        # Let's just use a known list or generate it properly.
        
        # Generation of PG(2,5) points
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero entry 1
                    # Since field is 5, we can divide by first non-zero
                    idx = -1
                    for i in range(3):
                        if (x, y, z)[i] != 0:
                            idx = i
                            break
                    val = (x, y, z)[idx]
                    inv_val = pow(val, 3, 5) # Fermat little theorem: a^(p-2) = a^-1
                    nx = (x * inv_val) % 5
                    ny = (y * inv_val) % 5
                    nz = (z * inv_val) % 5
                    points.append((nx, ny, nz))
                    
        # Remove duplicates (should be 31 unique normalized points)
        unique_points = list(set(points))
        # Map point to index
        point_to_idx = {p: i for i, p in enumerate(unique_points)}
        
        # Polarity condition: x*x' + y*y' + z*z' = 0 mod 5
        # This is a symmetric bilinear form.
        # Self-orthogonal points (absolute points) are not connected to themselves (simple graph).
        # u ~ v if dot(u, v) = 0
        
        for i in range(len(unique_points)):
            for j in range(i + 1, len(unique_points)):
                p1 = unique_points[i]
                p2 = unique_points[j]
                dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return {'n': n, 'edges': edges}

    return get_polarity_graph_q5()
