"""Strong algorithm for C4-free edge set in Q7 using max independent set."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Enumerate all edges of Q7
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    m = len(edges)  # 448
    
    # Build conflict graph: two edges conflict if they form a 4-cycle
    # Edge e1 = (x, x^bit_i) and e2 = (y, y^bit_j) conflict if:
    # x ^ y = bit_i ^ bit_j
    # For each edge, find conflicting edges
    # Edge idx corresponds to (x, y) where y = x ^ (1<<i)
    # Conflicts with edges (x^(1<<j), y^(1<<j)) for j != i
    
    # conflict_mask[k] = bitmask of edges that conflict with edge k
    conflict_mask = [0] * m
    
    for idx, (x, y) in enumerate(edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            c_idx = edge_to_idx[(min(c, e), max(c, e))]
            conflict_mask[idx] |= (1 << c_idx)
    
    # comp_adj[k]: edges that can be in a clique with k (i.e., don't conflict with k)
    all_mask = (1 << m) - 1
    comp_adj = [all_mask ^ conflict_mask[k] ^ (1 << k) for k in range(m)]
    
    # Greedy initialization to get a good lower bound
    def greedy_max_independent_set():
        selected = 0
        candidates = all_mask
        while candidates:
            # Pick vertex with max degree in remaining conflict graph? 
            # Or just first one. For speed, pick first.
            lowbit = candidates & (-candidates)
            v = lowbit.bit_length() - 1
            selected |= lowbit
            # Remove v and its neighbors (conflicts) from candidates
            # In independent set, if we pick v, we cannot pick any neighbor in conflict graph.
            # Neighbors in conflict graph are bits in conflict_mask[v].
            candidates &= ~(conflict_mask[v] | lowbit)
        return selected.bit_count()
    
    best_size = greedy_max_independent_set()
    best_clique_mask = 0 # We store the mask of the best clique found
    
    # Tomita's Maximum Clique Algorithm
    # We work on the complement graph (comp_adj).
    # A clique in comp_adj is an independent set in conflict graph.
    
    def greedy_color(P):
        """
        Greedy coloring of the subgraph induced by P in comp_adj.
        Returns: 
          colors: list of bitmasks, where colors[i] is the set of vertices with color i+1.
        """
        color_classes = []
        remaining = P
        while remaining:
            # Pick a vertex to start a new color class
            # Heuristic: pick vertex with highest degree in remaining? 
            # For speed, pick lowest set bit.
            lowbit = remaining & (-remaining)
            v = lowbit.bit_length() - 1
            
            color_class = 0
            avail = remaining
            while avail:
                # Pick a vertex from avail
                lb = avail & (-avail)
                u = lb.bit_length() - 1
                color_class |= lb
                # Remove u and its neighbors in comp_adj from avail
                # If u is in this color class, no other vertex in this class can be adjacent to u in comp_adj.
                avail &= ~(comp_adj[u] | lb)
            
            color_classes.append(color_class)
            remaining &= ~color_class
        
        return color_classes

    def max_clique_rec(P, current_clique_mask, depth):
        nonlocal best_size, best_clique_mask
        
        # Pruning
        if depth + P.bit_count() <= best_size:
            return
        
        if P == 0:
            if depth > best_size:
                best_size = depth
                best_clique_mask = current_clique_mask
            return
        
        # Greedy coloring
        color_classes = greedy_color(P)
        max_color = len(color_classes)
        
        if depth + max_color <= best_size:
            return
        
        # Process colors in reverse order (highest color first)
        # Vertices with higher colors are "later" in the ordering
        for c in range(max_color - 1, -1, -1):
            if depth + 1 + c <= best_size:
                break
            
            cc = color_classes[c]
            # Iterate through vertices in this color class
            # We need to process them one by one, updating P
            # Standard Tomita: 
            # for v in color_class[c]:
            #    branch on v
            #    P = P & ~ (1<<v)  # remove v from P for next iterations in this color class
            #    # Note: The new P for recursion is P_new = P & comp_adj[v] & (union of color classes > c)
            
            # To optimize, we can extract the list of vertices in cc
            verts = []
            tmp = cc
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                verts.append(v)
                tmp ^= lb
            
            for v in verts:
                if depth + 1 + c <= best_size:
                    break
                
                # New candidate set: 
                # 1. Must be neighbor of v in comp_adj
                # 2. Must be in P
                # 3. Must have color > c (i.e., be in color_classes[c+1] ... color_classes[max_color-1])
                
                # Construct mask for colors > c
                # This is expensive to compute every time. 
                # Alternative: Since we are iterating v in cc, and we remove v from P, 
                # we can just use P & comp_adj[v] and rely on the coloring bound?
                # No, the bound `depth + 1 + c` assumes that the remaining candidates 
                # can only form a clique of size at most c (if we restricted to higher colors).
                # If we don't restrict, the bound is invalid.
                
                # Efficient way to restrict to higher colors:
                # higher_colors_mask = 0
                # for k in range(c+1, max_color):
                #     higher_colors_mask |= color_classes[k]
                
                # We can precompute suffix sums of color classes? 
                # Or simply: 
                # P_new = P & comp_adj[v]
                # But we must ensure we don't pick vertices with color <= c in the recursive call?
                # Actually, the recursive call will re-color. The bound `depth + 1 + c` is valid ONLY IF 
                # the vertices in P_new are restricted to those with color > c.
                
                # Let's construct the mask for colors > c
                # Since max_color is small (usually < 20), this loop is cheap.
                higher_mask = 0
                for k in range(c + 1, max_color):
                    higher_mask |= color_classes[k]
                
                P_new = P & comp_adj[v] & higher_mask
                
                max_clique_rec(P_new, current_clique_mask | (1 << v), depth + 1)
                
                # Remove v from P for the next vertex in this color class
                P ^= (1 << v)

    start_P = all_mask
    max_clique_rec(start_P, 0, 0)
    
    # Extract edges from best_clique_mask
    result = []
    mask = best_clique_mask
    while mask:
        lowbit = mask & (-mask)
        idx = lowbit.bit_length() - 1
        x, y = edges[idx]
        result.append([x, y])
        mask ^= lowbit
        
    return result

# EVOLVE-BLOCK-END
