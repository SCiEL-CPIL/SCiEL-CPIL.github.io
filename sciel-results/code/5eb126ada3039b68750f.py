import time
from math import gcd

def construct():
    N = 512
    deadline = time.time() + 80  # Safety margin

    # 1. Build the compatibility graph using bitsets
    # adj_mask[v] is a bitmask of all u != v such that lcm(v, u) <= N
    adj_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= N:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
    
    # Precompute bit positions for vertices 1..N
    # Note: Python integers are unbounded, so 1 << 512 is fine.
    # We will use indices 1..N.

    best_clique = []
    best_size = 0

    # Tomita's algorithm for Maximum Clique
    # P: set of candidate vertices (bitmask)
    # X: set of vertices that have been processed and are not in the current path (not strictly needed for size, but for reconstruction if we wanted the set)
    # We just need the size, but the task requires returning the subset. 
    # Reconstructing the set from Tomita is slightly tricky if we only track size.
    # We can track the path.
    
    # Since N=512, we can store the path as a list.
    
    def expand(P, X, path):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(path) > best_size:
                best_size = len(path)
                best_clique = list(path)
            return
            
        # Greedy Coloring to get an upper bound
        # We need to color the vertices in P.
        # The vertices are represented by bits in P.
        # We extract the list of vertices in P.
        verts = []
        p = P
        while p:
            # Get lowest set bit
            lsb = p & -p
            v = lsb.bit_length() - 1
            verts.append(v)
            p ^= lsb
            
        if not verts:
            return
            
        # Order vertices by degree in P (descending) for better coloring
        # deg in P: number of neighbors in P
        # We can compute this using bit operations
        degs = []
        for v in verts:
            # Neighbors of v in P
            nbrs_in_p = adj_mask[v] & P
            degs.append((nbrs_in_p.bit_count(), v))
        degs.sort(key=lambda x: -x[0])
        verts = [v for _, v in degs]
        
        # Greedy Coloring
        # colors[v] = color index
        # We want to assign the smallest possible color to each vertex
        # such that no two adjacent vertices have the same color.
        # The max color used + 1 is an upper bound on the clique size in P.
        
        color = {}
        # We process vertices in the order defined by 'verts' (which is degree-sorted)
        # But standard Tomita processes in reverse degree order for coloring to get a tight bound?
        # Actually, the standard approach:
        # 1. Order vertices by degree.
        # 2. Color greedily.
        # 3. The max color is the bound.
        # 4. Branch on vertices in reverse color order.
        
        # Let's use a simpler greedy coloring:
        # For each vertex v in verts:
        #   Find the smallest color c not used by any neighbor of v that has already been colored.
        
        used_colors = [set() for _ in range(N + 1)] # used_colors[c] contains vertices with color c? No, that's not efficient.
        # Better: For each vertex, keep a bitmask of colors used by its colored neighbors?
        # Or simply: for vertex v, check colors of neighbors.
        
        # Since N is small (512), we can just iterate.
        # To speed up, we can use the property that if a vertex is connected to all previous vertices of color c, it can't take color c.
        
        color_assign = {}
        max_color = 0
        for v in verts:
            # Check neighbors of v that are already colored
            # A neighbor u is colored if u is in color_assign
            # We need to know which colors are "blocked" by colored neighbors.
            blocked = set()
            for u in verts:
                if u in color_assign:
                    # Check if u is a neighbor of v
                    if (adj_mask[v] >> u) & 1:
                        blocked.add(color_assign[u])
            
            c = 0
            while c in blocked:
                c += 1
            color_assign[v] = c
            if c > max_color:
                max_color = c
                
        # Upper bound: max_color + 1
        if len(path) + max_color + 1 <= best_size:
            return

        # Branch
        # We should process vertices in decreasing order of their color.
        # Vertices with higher color are "more constrained" or "later" in the coloring order.
        # Standard Tomita:
        # For i from max_color down to 0:
        #   For each vertex v with color i:
        #       Expand(P & adj_mask[v], X | (P & ~adj_mask[v]), path + [v])
        #       P = P & ~ (1 << v)
        #       X = X | (1 << v)
        
        # We need to group vertices by color.
        # colors_to_verts = {c: [v1, v2, ...]}
        colors_to_verts = {}
        for v in verts:
            c = color_assign[v]
            if c not in colors_to_verts:
                colors_to_verts[c] = []
            colors_to_verts[c].append(v)
            
        # Process colors in descending order
        for c in range(max_color, -1, -1):
            if c not in colors_to_verts:
                continue
            for v in colors_to_verts[c]:
                if time.time() > deadline:
                    return
                
                # Check if v is still in P
                if (P >> v) & 1 == 0:
                    continue
                    
                # New P: neighbors of v in P
                new_P = P & adj_mask[v]
                # New X: vertices in P that are not neighbors of v (plus old X, but X doesn't affect size bound directly in this simplified view, 
                # though strictly X is needed for correctness of the "remaining" set if we were doing independent set, 
                # for clique, P & adj_mask[v] is sufficient for the next step's candidates, 
                # provided we remove v from P for the next iteration of the loop).
                
                # Recurse
                expand(new_P, X, path + [v])
                
                # Remove v from P for subsequent iterations in this level
                P &= ~(1 << v)
                
                # Update bound check for remaining vertices in this color and lower colors?
                # The bound `len(path) + max_color + 1` was for the whole P.
                # As we remove vertices, the bound decreases.
                # A tighter bound: len(path) + (number of colors remaining) is not quite right.
                # The standard pruning is: if len(path) + (current max color of remaining P) <= best_size, break.
                # Since we process in descending color order, the max color of the remaining P is at most c.
                # So if len(path) + c + 1 <= best_size, we can break out of the color loop?
                # Actually, if we are at color c, the remaining vertices have colors <= c.
                # So the max clique size we can add is at most c + 1? No, it's at most the number of colors remaining.
                # If we have colors c, c-1, ..., 0 remaining, the max clique size is bounded by c+1.
                # So if len(path) + c + 1 <= best_size, we can stop processing this color and lower ones.
                if len(path) + c + 1 <= best_size:
                    break
            if len(path) + c + 1 <= best_size:
                break

    # Initial call
    # P is all vertices 1..N
    P = 0
    for i in range(1, N + 1):
        P |= (1 << i)
        
    expand(P, 0, [])
    
    # If no clique found (should not happen), return a valid small set
    if not best_clique:
        best_clique = [1]
        
    return best_clique
