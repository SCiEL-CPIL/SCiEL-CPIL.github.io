import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    # Greedy coloring for bound in Max Clique (MIS) problem
    # We solve MIS by finding Max Clique in the complement graph.
    # comp_adj[v] = vertices in cand_mask that are NOT adjacent to v in original graph (and not v itself)
    comp_adj = [0] * N
    for v in range(N):
        if not forbidden[v]:
            comp_adj[v] = cand_mask & ~adj[v] & ~(1 << v)
        else:
            comp_adj[v] = 0

    # Greedy coloring of a set of vertices in the complement graph
    # Returns a list of (vertex, color) pairs, and the max color used
    def greedy_color(vertices_mask):
        # vertices_mask is a bitmask of vertices to color
        # We color them in order of decreasing degree in the complement graph
        # For efficiency, we'll use a simple approach:
        # 1. Extract vertices
        # 2. Sort by degree in complement (within the set)
        # 3. Assign colors greedily
        
        verts = []
        m = vertices_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        if not verts:
            return [], 0
        
        # Sort by degree in complement graph (within the set), descending
        # Degree of v in complement within set = popcount(comp_adj[v] & vertices_mask)
        verts.sort(key=lambda v: (comp_adj[v] & vertices_mask).bit_count(), reverse=True)
        
        colors = {}
        color_classes = []  # list of bitmasks, one per color
        
        for v in verts:
            # Find the smallest color class that doesn't contain a neighbor of v in complement
            # In complement graph, v can share a color with u if they are NOT adjacent in complement
            # i.e., u is NOT in comp_adj[v]
            # So v can be in color class c if (color_classes[c] & comp_adj[v]) == 0
            
            placed = False
            for c in range(len(color_classes)):
                if (color_classes[c] & comp_adj[v]) == 0:
                    color_classes[c] |= (1 << v)
                    colors[v] = c + 1
                    placed = True
                    break
            
            if not placed:
                color_classes.append(1 << v)
                colors[v] = len(color_classes)
        
        max_color = len(color_classes)
        return colors, max_color

    # Branch and Bound for Maximum Clique in complement graph (= Maximum Independent Set in original)
    # Using Tomita-style algorithm with greedy coloring bound
    
    best_size = [0]
    best_set = [0]
    node_count = [0]
    
    def max_clique_bnb(P, current_size, current_set):
        # P is a bitmask of candidate vertices
        # current_size is the size of the clique found so far
        # current_set is the bitmask of the clique found so far
        
        if time.time() - start_time > DEADLINE:
            return
        
        if P == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set
            return
        
        # Bound: greedy coloring of P in complement graph
        colors, max_color = greedy_color(P)
        
        if current_size + max_color <= best_size[0]:
            return
        
        # Process vertices in reverse order of their color assignment
        # (higher color vertices first, as they are more promising)
        # Extract vertices from P
        verts = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Sort by color descending (higher color = more promising)
        verts.sort(key=lambda v: colors.get(v, 0), reverse=True)
        
        # Process each vertex
        for v in verts:
            if time.time() - start_time > DEADLINE:
                return
            
            node_count[0] += 1
            
            # Prune: if current_size + 1 + (number of vertices with color > colors[v]) <= best_size[0]
            # Actually, the standard Tomita bound: if current_size + colors[v] <= best_size[0], skip
            if current_size + colors[v] <= best_size[0]:
                break
            
            # Include v in the clique
            new_set = current_set | (1 << v)
            # New candidates are vertices in P that are neighbors of v in complement graph
            # and have color >= colors[v] (to maintain the ordering property)
            # Actually, we need vertices in P that are adjacent to v in complement
            # and appear after v in our ordering (to avoid duplicates)
            # In Tomita's algorithm, we only consider vertices that come after v in the sorted order
            
            # Find vertices in P that are neighbors of v in complement and have index > v's position
            # For simplicity, we'll just intersect P with comp_adj[v] and remove v
            # But we need to be careful about the ordering to avoid duplicates
            
            # Standard approach: new_P = P & comp_adj[v] & (vertices after v in sorted order)
            # To get "vertices after v", we can just remove v and all vertices before it
            # But since we're iterating in order, we can just use P & comp_adj[v] and remove v
            
            new_P = P & comp_adj[v]
            # Remove v from new_P (though it shouldn't be there since comp_adj[v] doesn't include v)
            new_P &= ~(1 << v)
            
            # Also, we should only consider vertices that come after v in our sorted order
            # to avoid exploring the same clique multiple times
            # This is the key to Tomita's algorithm efficiency
            
            # For now, let's use a simpler approach: just recurse
            max_clique_bnb(new_P, current_size + 1, new_set)
            
            # Remove v from P for subsequent iterations
            P &= ~(1 << v)
            
            # Update bound: max_color decreases as we remove vertices
            # We could recompute, but for simplicity, we'll just continue
            # A tighter bound would be to recompute greedy_color on the remaining P

    # Start the search
    max_clique_bnb(cand_mask, 0, 0)

    # Convert best_set to list of values (1-based)
    result = []
    m = best_set[0]
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb

    return sorted(result)

# EVOLVE-BLOCK-END
