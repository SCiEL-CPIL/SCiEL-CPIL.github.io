import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) creates a K3,3 subgraph.
        Pure bitmask operations, no list extraction.
        """
        c_y_mask = adj[u] & ~(1 << v)
        c_x_mask = adj[v] & ~(1 << u)
        
        # Need at least 2 bits in c_y_mask
        if not (c_y_mask & (c_y_mask - 1)):
            return True
        
        # Iterate over pairs in c_y_mask using bit tricks
        mask = c_y_mask
        while mask:
            lsb1 = mask & -mask
            y1 = lsb1.bit_length() - 1
            mask2 = mask ^ lsb1
            mask = mask2
            while mask2:
                lsb2 = mask2 & -mask2
                y2 = lsb2.bit_length() - 1
                mask2 = mask2 ^ lsb2
                
                potential_x = (adj[y1] & adj[y2]) & c_x_mask
                if potential_x & (potential_x - 1):
                    return False
        return True

    def adj_to_edges(adj):
        edges = []
        for i in range_n:
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def get_missing_edges(adj):
        """Return list of (u,v) pairs that are not in adj."""
        missing = []
        for u in range_n:
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        return missing

    best_edges = []
    best_count = 0
    best_adj = [0] * n

    # --- Structured initial construction: polarity graph of PG(2,4) ---
    # PG(2,4) has 21 points. We use 20 of them.
    # The polarity graph has edges between points that are "orthogonal" under a polarity.
    # This gives a strongly regular graph with parameters (21, 10, 4, 5) or similar.
    # We'll construct it explicitly.
    
    def build_polarity_graph():
        """Construct a K3,3-free graph from polarity of projective plane of order 4."""
        # Points of PG(2,4) as 3-tuples over GF(4), up to scalar.
        # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
        # We represent elements as: 0, 1, 2 (=alpha), 3 (=alpha+1)
        # Multiplication table for GF(4):
        # 0*x=0, 1*x=x, 2*2=3, 2*3=1, 3*3=2
        
        def gf4_add(a, b):
            return a ^ b  # XOR for GF(4) addition (since char 2)
        
        def gf4_mul(a, b):
            if a == 0 or b == 0:
                return 0
            if a == 1:
                return b
            if b == 1:
                return a
            # a,b in {2,3}
            lookup = {(2,2): 3, (2,3): 1, (3,2): 1, (3,3): 2}
            return lookup[(a, b)]
        
        def gf4_inv(a):
            if a == 0:
                return 0
            if a == 1:
                return 1
            if a == 2:
                return 3
            if a == 3:
                return 2
        
        # Generate all 21 points of PG(2,4)
        # Each point is a 1-dim subspace, represented by a nonzero vector (x,y,z) normalized
        # We normalize so that the first nonzero coordinate is 1.
        
        points = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: find first nonzero, scale so it's 1
                    if x != 0:
                        inv_x = gf4_inv(x)
                        pt = (1, gf4_mul(inv_x, y), gf4_mul(inv_x, z))
                    elif y != 0:
                        inv_y = gf4_inv(y)
                        pt = (0, 1, gf4_mul(inv_y, z))
                    else:
                        pt = (0, 0, 1)
                    
                    # Check if already in points
                    found = False
                    for p in points:
                        if p == pt:
                            found = True
                            break
                    if not found:
                        points.append(pt)
        
        # Should have 21 points
        # Define a polarity: map point (x,y,z) to line xz + y^2 = 0 (or similar)
        # Two points are adjacent if one lies on the polar of the other.
        # Polar of point (a,b,c): line ax + by + cz = 0 where we use a symmetric bilinear form.
        # For a symmetric polarity, use bilinear form B(p,q) = p0*q2 + p2*q0 + p1*q1 = 0
        
        def is_orthogonal(p, q):
            # B(p,q) = p0*q2 + p2*q0 + p1*q1 in GF(4)
            val = gf4_add(gf4_add(gf4_mul(p[0], q[2]), gf4_mul(p[2], q[0])), gf4_mul(p[1], q[1]))
            return val == 0
        
        # Build adjacency for 21 points
        adj21 = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                if is_orthogonal(points[i], points[j]):
                    adj21[i] |= (1 << j)
                    adj21[j] |= (1 << i)
        
        # Use first 20 points (drop point 20)
        adj20 = [0] * 20
        for i in range(20):
            for j in range(20):
                if i != j and (adj21[i] >> j) & 1:
                    adj20[i] |= (1 << j)
        
        return adj20

    # Build the initial graph
    init_adj = build_polarity_graph()
    init_edges = adj_to_edges(init_adj)
    init_count = len(init_edges)
    
    # Verify K3,3-free
    def is_k33_free(adj):
        for i in range_n:
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if bin(common).count('1') >= 3:
                        return False
        return True
    
    if not is_k33_free(init_adj):
        # Fallback: start from scratch
        init_adj = [0] * n
        init_edges = []
        init_count = 0
    
    # Use the structured graph as initial best
    if init_count > best_count:
        best_count = init_count
        best_adj = init_adj.copy()
        best_edges = init_edges.copy()

    # Main search loop
    while time.time() < deadline:
        # Start from best_adj or random
        if random.random() < 0.5 and best_count > 0:
            adj = best_adj.copy()
            current_edges = adj_to_edges(adj)
        else:
            # Random greedy
            shuffled = all_edges.copy()
            random.shuffle(shuffled)
            adj = [0] * n
            current_edges = []
            for u, v in shuffled:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_addition(adj, u, v):
                    current_edges.append((u, v))
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # Local search: edge swap optimization
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            if not current_edges:
                break
            
            # Pick a random edge to remove
            edge_idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[edge_idx]
            
            # Remove it
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(edge_idx)
            
            # Try to add missing edges (only those involving neighbors of removed edge endpoints)
            # For efficiency, try all missing edges but use fast bitmask check
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            
            added_any = False
            for u, v in missing:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid_addition(adj, u, v):
                    current_edges.append((u, v))
                    added_any = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if added_any:
                improved = True
            else:
                # Revert
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                current_edges.append((u_rem, v_rem))

        # Update best
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_adj = adj.copy()
            best_edges = adj_to_edges(best_adj)

    return best_edges
