import math

def construct():
    N = 512
    limit = 512
    
    # Build valid_adj: adj[i] has bit j set if lcm(i, j) <= 512
    # We use 0-indexed bits for i in 0..511 representing values 1..512
    adj = [0] * (N)
    for i in range(N):
        val_i = i + 1
        for j in range(i + 1, N):
            val_j = j + 1
            g = math.gcd(val_i, val_j)
            l = (val_i // g) * val_j
            if l <= limit:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_size = 0
    best_set_mask = 0
    
    def greedy_color(P_mask):
        # Returns a list of colors for each vertex in P_mask
        # and the maximum color used (upper bound on clique size)
        # We use a greedy coloring on the subgraph induced by P_mask
        # Note: For Max Clique in G, we color G. A clique must have distinct colors.
        # So max_color is an upper bound on clique size.
        
        if P_mask == 0:
            return [], 0
            
        # Extract vertices
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in P_mask descending to get a tighter coloring
        # This helps minimize the number of colors used
        verts.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)
        
        colors = {} # vertex index -> color
        color_classes = [] # list of bitmasks for each color
        
        for v in verts:
            v_bit = 1 << v
            placed = False
            # Try to assign the lowest possible color
            for c_idx, c_mask in enumerate(color_classes):
                # v can share color c if it is NOT adjacent to any vertex in c_mask
                # In the Valid Graph, adjacent means compatible.
                # For Coloring the Valid Graph to bound Clique:
                # A clique consists of mutually adjacent vertices.
                # If we color the graph such that adjacent vertices have different colors,
                # then a clique must have all distinct colors.
                # So, v can share color c if v is NOT adjacent to ANY vertex in c_mask?
                # No, wait.
                # Standard Coloring for Max Clique Bound:
                # We color the graph G. If u and v are adjacent in G, they must have different colors.
                # Therefore, a clique (set of mutually adjacent vertices) must have all distinct colors.
                # So, if we find a coloring with k colors, the max clique size <= k.
                # To assign a color to v, we check if v is adjacent to any vertex in color class c.
                # If v is adjacent to ANY vertex in c_mask, it CANNOT take color c.
                # If v is NOT adjacent to ANY vertex in c_mask, it CAN take color c?
                # No. If v is not adjacent to u (where u is in c), they can share a color.
                # So, v can take color c if (adj[v] & c_mask) == 0.
                
                if (adj[v] & c_mask) == 0:
                    colors[v] = c_idx
                    color_classes[c_idx] |= v_bit
                    placed = True
                    break
            if not placed:
                colors[v] = len(color_classes)
                color_classes.append(v_bit)
                
        max_color = len(color_classes)
        return colors, max_color

    def expand(P_mask, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_set_mask
            
        if P_mask == 0:
            return
            
        # Compute coloring bound
        colors, max_color = greedy_color(P_mask)
        
        if current_size + max_color <= best_size:
            return
            
        # We want to process vertices in a way that allows early pruning.
        # In Tomita's algorithm, we process vertices in reverse order of their color.
        # Vertices with higher colors are "more constrained" or rather, the bound decreases as we remove them?
        # Actually, the standard trick is:
        # Sort vertices by color (ascending).
        # Iterate through them.
        # If current_size + color[v] <= best_size, we can prune?
        # Let's stick to the simpler but effective heuristic:
        # Iterate through vertices in P_mask.
        # To make pruning effective, we should process vertices that are likely to be in the max clique first?
        # Or rather, we want to reduce P_mask quickly.
        
        # Let's iterate through vertices sorted by their color in descending order?
        # Actually, if we process a vertex v with color c, and we don't pick it, we remove it.
        # The bound for the remaining set might decrease.
        
        # A robust approach:
        # Get vertices in P_mask.
        # Sort them by (color, degree) descending?
        # Let's just iterate through all vertices in P_mask.
        
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by color descending, then degree descending
        verts.sort(key=lambda v: (colors.get(v, 0), (adj[v] & P_mask).bit_count()), reverse=True)
        
        for v in verts:
            # Pruning: If current_size + color[v] <= best_size, we can skip?
            # Not exactly. The color is an upper bound for the clique in the *original* P_mask.
            # As we remove vertices, the bound might be lower.
            # However, if we process in order of decreasing color, and we haven't found a better solution,
            # we can stop if current_size + colors[v] <= best_size.
            
            if current_size + colors.get(v, 0) <= best_size:
                continue
                
            v_bit = 1 << v
            
            # Remove v from P_mask for subsequent iterations in this loop
            # This is the "exclude v" branch
            P_mask_excl = P_mask & ~v_bit
            
            # Include v
            new_P = P_mask_excl & adj[v]
            expand(new_P, current_size + 1, current_set_mask | v_bit)
            
            # Update P_mask for the next vertex in the loop
            # The loop variable `verts` is fixed, but we need to ensure we are working with the updated set
            # for the "exclude" part?
            # In standard B&B for Max Clique:
            # We iterate through a list of candidates.
            # For each candidate v:
            #   1. Recurse with v included.
            #   2. Remove v from the candidate set for the remaining iterations.
            
            # My loop above uses a fixed `verts` list. I need to update the mask used for the next iterations.
            # Let's restructure the loop to handle the mask update correctly.
            
            # Actually, the `P_mask` variable is local to this call.
            # I will modify the loop to explicitly manage the "remaining candidates".
            
            # Let's restart the loop logic to be clearer.
            pass # This block was just thought process, the actual code below will be cleaner.

    # Re-implementing expand with correct mask management
    def expand(P_mask, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_set_mask
            
        if P_mask == 0:
            return
            
        colors, max_color = greedy_color(P_mask)
        
        if current_size + max_color <= best_size:
            return
            
        # Prepare list of vertices to try
        # We will iterate through them.
        # We need to remove them from P_mask as we go (the "exclude" branch is implicit by not including them in the next steps of the loop, but we must ensure the "include" branch uses the correct neighbors).
        
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by color descending to allow early termination if bound drops
        verts.sort(key=lambda v: colors.get(v, 0), reverse=True)
        
        # We need to maintain a "remaining" mask for the loop
        remaining = P_mask
        
        for v in verts:
            # Check if v is still in remaining (it should be, as we only remove after processing)
            v_bit = 1 << v
            if not (remaining & v_bit):
                continue
                
            # Pruning check
            if current_size + colors.get(v, 0) <= best_size:
                # Since we sorted by color descending, all subsequent vertices have <= color
                # So we can break?
                # Yes, if the best possible addition from this vertex and any later vertex
                # is <= best_size, we can stop.
                # However, the color bound `colors[v]` is for the *initial* P_mask.
                # As we remove vertices, the bound for the remaining set decreases.
                # But `colors[v]` is an upper bound on the clique size in the subgraph induced by vertices with color >= colors[v]?
                # Not exactly.
                # Let's stick to the simple check: if current_size + colors[v] <= best_size, skip/break.
                # This is a standard optimization in Tomita's algorithm.
                break
                
            # Branch 1: Include v
            new_P = remaining & adj[v]
            expand(new_P, current_size + 1, current_set_mask | v_bit)
            
            # Branch 2: Exclude v
            # Remove v from remaining for the next iterations
            remaining &= ~v_bit

    all_vertices = (1 << N) - 1
    expand(all_vertices, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
