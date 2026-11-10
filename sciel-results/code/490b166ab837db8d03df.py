import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s margin for safety
    
    N = 256
    # adj[i] is a bitmask of neighbors of i in the complement graph (allowed pairs)
    # i.e., adj[i] has bit j set if (i+j) does not divide i*j
    # Note: We are looking for a Maximum Clique in this graph.
    adj = [0] * (N + 1)
    
    # Precompute adjacency
    for u in range(1, N + 1):
        mask = 0
        for v in range(1, N + 1):
            if u == v:
                continue
            s = u + v
            prod = u * v
            if prod % s != 0:
                mask |= (1 << (v - 1))
        adj[u] = mask
        
    # Calculate degrees
    degrees = [0] * (N + 1)
    for u in range(1, N + 1):
        degrees[u] = bin(adj[u]).count('1')
        
    # Order vertices by descending degree
    # This helps in coloring bound
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map vertex value to its index in the sorted list?
    # Actually, for Tomita, we usually keep the original bit positions but iterate
    # in the sorted order.
    # Let's create a list of vertices in sorted order: v_list
    v_list = vertices
    
    best_size = 0
    best_clique = []
    
    # Greedy Coloring of a set of vertices (bitmask)
    # Returns a list of colors for each vertex in the set, and the max color.
    # The color assigned to a vertex v is an upper bound on the size of a clique
    # containing v and any vertices with lower colors.
    # We return an array `colors` where colors[i] is the color of the i-th vertex
    # in the ordered list of active vertices.
    
    def color_bound(P_mask):
        # P_mask is a bitmask of candidates
        # We color the subgraph induced by P_mask.
        # We process vertices in the order defined by v_list (highest degree first).
        # Wait, standard Tomita colors in reverse order of the search?
        # No, we color the candidates P.
        # Let's extract the list of active vertices in P_mask, sorted by v_list order.
        # v_list is sorted by degree desc.
        
        active = []
        for v in v_list:
            if (P_mask >> (v-1)) & 1:
                active.append(v)
                
        if not active:
            return [], 0
            
        colors = {}
        max_c = 0
        
        # Greedy coloring:
        # For each vertex v in active (in the order of v_list),
        # assign the smallest color not used by its neighbors in active that have already been colored.
        # Since we process in v_list order, and v_list is static, we can just do this.
        # However, to get a tight bound, the order matters.
        # Standard practice: Color in the order they appear in the search queue.
        
        # Let's use a simple approach:
        # colors[v] = c
        # used_colors_in_active = set of colors used by neighbors of v in active
        
        # To be efficient, we can maintain for each color c, a bitmask of vertices with color c.
        # But N is small, so set/dict is fine.
        
        color_map = {}
        # We want to minimize max color.
        # Iterate through active vertices.
        for v in active:
            # Find neighbors of v in active
            # n_mask = adj[v] & P_mask
            # We need to check which colors are used by these neighbors.
            # Since we are coloring sequentially, we only consider neighbors that are already colored.
            # In the `active` list, vertices before v are already colored.
            
            used = set()
            # Iterate through neighbors of v in active
            # Instead of iterating all bits in adj[v], we can just check the color_map of neighbors.
            # But we don't have a fast neighbor iterator.
            # Alternative: For each color c, check if any neighbor of v has color c.
            
            # Let's try:
            # For c in range(max_c + 1):
            #   if any neighbor of v in active has color c: used.add(c)
            
            # To make this fast, we can store `vertices_by_color` as a list of sets or bitmasks.
            # Let's use bitmasks for colors.
            # color_masks[c] = bitmask of vertices with color c
            
            # Re-initialize for each call? No, this is inside the recursion.
            # Let's just do the set approach, it's O(N^2) per coloring, but N<=256.
            # 256^2 = 65k ops. Acceptable if number of nodes is small.
            
            # Actually, let's optimize:
            # We can check if (adj[v] & color_mask[c]) != 0.
            
            # We need to build color_masks as we go.
            # Let's restart the logic inside the function to be self-contained and simple.
            
            pass # Placeholder, will implement below
            
        # Let's implement a cleaner version of color_bound
        # that returns the color of each vertex in the active list,
        # such that color[v] is the upper bound for cliques starting at v.
        
        # Revised color_bound:
        # 1. Get active vertices in order of v_list.
        # 2. Initialize color_masks = [0]
        # 3. For each v in active:
        #    - Determine smallest color c such that (adj[v] & color_masks[c]) == 0
        #    - If no such c <= len(color_masks), append 0 to color_masks and c = len(color_masks)-1
        #    - Set bit v in color_masks[c]
        #    - Record color[v] = c+1 (1-indexed)
        # 4. Return list of colors corresponding to active vertices, and max color.
        
        active_list = [v for v in v_list if (P_mask >> (v-1)) & 1]
        if not active_list:
            return [], 0
            
        color_masks = [0] # Index 0 is color 1
        vertex_colors = []
        
        for v in active_list:
            v_bit = 1 << (v-1)
            assigned = False
            for c_idx in range(len(color_masks)):
                # Check if v has any neighbor with this color
                if (adj[v] & color_masks[c_idx]) == 0:
                    color_masks[c_idx] |= v_bit
                    vertex_colors.append(c_idx + 1)
                    assigned = True
                    break
            if not assigned:
                color_masks.append(v_bit)
                vertex_colors.append(len(color_masks))
                
        max_c = len(color_masks)
        return vertex_colors, max_c

    # Expansion function
    # R: list of vertices in current clique
    # P_mask: bitmask of candidates
    # colors: list of colors for vertices in P_mask (in order of v_list intersection with P_mask)
    # This is tricky to manage state. 
    # Standard Tomita passes the color array.
    
    def expand(R, P_mask, colors):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # The colors array corresponds to the vertices in P_mask in the order of v_list.
        # Let's extract the list of vertices in P_mask in v_list order.
        # This is expensive if done at every node.
        # Better: Pass the list of vertices alongside P_mask?
        # Or, since v_list is fixed, we can iterate v_list and check P_mask.
        
        # Let's restructure:
        # Instead of passing colors, we recompute or pass the list of (vertex, color) pairs.
        # Given the complexity, let's use a simpler B&B with the bound from coloring
        # computed at the entry of expand.
        
        # To avoid recomputing active list every time, we can pass it.
        # But let's stick to the bitmask and recompute active list if needed.
        # For N=256, iterating v_list (256 items) is fast.
        
        # Get active vertices and their colors
        # Note: The `colors` list from color_bound corresponds to the active vertices in v_list order.
        
        active_vertices = []
        for i, v in enumerate(v_list):
            if (P_mask >> (v-1)) & 1:
                active_vertices.append((v, colors[i]))
                
        # If active_vertices is empty, P_mask was 0, handled above.
        
        # Bound check:
        # The maximum possible clique size is len(R) + max_color_in_P.
        # The colors are 1-indexed.
        if not active_vertices:
            return
            
        max_color = max(c for _, c in active_vertices)
        if len(R) + max_color <= best_size:
            return
            
        # Branching:
        # We iterate through active_vertices in reverse order of color?
        # Or just in the order they appear?
        # Standard Tomita:
        # For each vertex v in P (in order of increasing color? or just order):
        #   If len(R) + color[v] <= best_size: break (since remaining have <= color)
        #   Include v, recurse with P' = P & adj[v]
        
        # We need to process vertices such that we can prune based on color.
        # The colors in `active_vertices` are assigned such that if vertex v has color c,
        # any clique containing v can have size at most c (relative to P).
        # So we should iterate v in order of decreasing color to prune early?
        # Actually, if we sort by color descending, and len(R) + color <= best, we can stop.
        
        # Let's sort active_vertices by color descending.
        active_vertices.sort(key=lambda x: x[1], reverse=True)
        
        for v, c in active_vertices:
            if len(R) + c <= best_size:
                break # Since colors are non-increasing, we can stop
                
            # Include v
            R.append(v)
            
            # New P: neighbors of v in current P
            new_P = P_mask & adj[v]
            new_P &= ~(1 << (v-1)) # Remove v itself (though adj[v] shouldn't have v)
            
            # Compute colors for new_P
            new_colors, _ = color_bound(new_P)
            
            expand(R, new_P, new_colors)
            
            R.pop()
            
            # Remove v from P for subsequent iterations (exclude v)
            P_mask &= ~(1 << (v-1))
            
            # Update colors?
            # The colors of remaining vertices might change?
            # No, the color bound is an upper bound. Removing a vertex doesn't increase the bound.
            # However, the `colors` list we have is for the original P.
            # When we remove v, the indices in `active_vertices` shift?
            # This implementation is getting messy with indices.
            
            # Let's restart the loop logic or just re-compute colors for the reduced P?
            # Re-computing is safer but slower.
            # Given N=256, and pruning is strong, let's re-compute colors for the reduced P
            # at the start of the next iteration? No, we are in a loop.
            
            # Alternative: Just call expand with the reduced P and let it re-compute colors?
            # No, expand takes colors as input.
            
            # Let's modify expand to compute colors if not provided?
            # Or just re-calculate colors for the new P_mask inside the loop before the next iteration?
            
            # Actually, the standard algorithm updates the color array in place.
            # For simplicity and correctness within time limits, let's just re-calculate
            # the colors for the remaining P_mask in the next iteration?
            # No, that's O(N) extra per node.
            
            # Let's try a different approach:
            # Don't pass colors. Compute them at the start of expand.
            
            pass # Break out of this complex loop structure and rewrite expand

    # Rewrite expand to be simpler: Compute colors at start.
    
    def expand_simple(R, P_mask):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Compute colors for P_mask
        colors, max_c = color_bound(P_mask)
        
        # If bound is too low, prune
        if len(R) + max_c <= best_size:
            return
            
        # Get active vertices with their colors
        # We need to map color index back to vertex.
        # color_bound returns colors in order of v_list.
        
        active = []
        idx = 0
        for v in v_list:
            if (P_mask >> (v-1)) & 1:
                active.append((v, colors[idx]))
                idx += 1
                
        # Sort by color descending to allow early break
        active.sort(key=lambda x: x[1], reverse=True)
        
        for v, c in active:
            if len(R) + c <= best_size:
                break
                
            R.append(v)
            
            new_P = P_mask & adj[v]
            new_P &= ~(1 << (v-1))
            
            expand_simple(R, new_P)
            
            R.pop()
            
            P_mask &= ~(1 << (v-1))
            
            # Note: We do not update `colors` here.
            # The bound `c` is an upper bound for the clique containing v.
            # For the next vertex, its color in the *original* coloring is still a valid upper bound
            # for a clique containing it (and not v, since we removed v from P).
            # So we can just continue with the existing active list and colors.
            # The condition `len(R) + c <= best_size` uses the color from the initial coloring of P.
            # This is valid.

    # Initial call
    init_P = (1 << N) - 1
    expand_simple([], init_P)
    
    if not best_clique:
        best_clique = [1]
        
    return best_clique
