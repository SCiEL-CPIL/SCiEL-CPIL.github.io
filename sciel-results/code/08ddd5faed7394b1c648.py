<<<<<<< SEARCH
def _find_valid_coloring(n, time_limit=10.0):
    """
    Find a valid 4-coloring of K_n using local search with C6-counting heuristic.
    """
    import random
    random.seed(42)
    
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    def count_c6s(colors):
        """Count total number of monochromatic C6s in the coloring."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        total = 0
        for c in range(4):
            # Count C6s in color class c
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    total += 1
        return total
    
    # Start with random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    current_c6s = count_c6s(colors)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    # Local search: flip random edges, keep if it reduces C6 count
    for iteration in range(100000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
            
        # Pick a random edge to flip
        edge_idx = random.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        colors[edge_idx] = new_color
        new_c6s = count_c6s(colors)
        
        if new_c6s <= best_c6s:
            best_c6s = new_c6s
            best_colors = colors[:]
        else:
            colors[edge_idx] = old_color
    
    if best_c6s == 0:
        return best_colors
    return None
=======
def _find_valid_coloring(n, time_limit=10.0):
    """
    Find a valid 4-coloring of K_n using local search with C6-counting heuristic.
    Uses incremental updates and targeted edge selection.
    """
    import random
    random.seed(42)
    
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    # Build edge index mapping
    edge_to_ij = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_ij.append((i, j))
    
    def build_adj(colors):
        """Build adjacency bitmasks for all 4 color classes."""
        adj = [[0] * n for _ in range(4)]
        for idx, (i, j) in enumerate(edge_to_ij):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        return adj
    
    def count_c6s_from_adj(adj):
        """Count total number of monochromatic C6s from precomputed adj."""
        total = 0
        for c in range(4):
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    total += 1
        return total
    
    def find_c6_edges(adj, colors):
        """Find edges that are part of monochromatic C6s."""
        c6_edges = set()
        for c in range(4):
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    # Found a C6: v0-v1-v2-v3-v4-v5-v0
                                    # All 6 edges are in this C6
                                    cycle_edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                                    for (a, b) in cycle_edges:
                                        if a > b:
                                            a, b = b, a
                                        # Find edge index
                                        idx = -1
                                        for k, (i, j) in enumerate(edge_to_ij):
                                            if i == a and j == b:
                                                idx = k
                                                break
                                        if idx >= 0:
                                            c6_edges.add(idx)
        return c6_edges
    
    # Start with random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = build_adj(colors)
    current_c6s = count_c6s_from_adj(adj)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    best_adj = [row[:] for row in adj]
    
    # Local search with targeted edge selection
    no_improve_count = 0
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
        
        # Get edges involved in C6s
        c6_edges = find_c6_edges(adj, colors)
        
        # Pick an edge: prefer ones in C6s, otherwise random
        if c6_edges and random.random() < 0.7:
            edge_idx = random.choice(list(c6_edges))
        else:
            edge_idx = random.randint(0, num_edges - 1)
        
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        if new_color == old_color:
            continue
        
        # Update adj incrementally
        i, j = edge_to_ij[edge_idx]
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        
        colors[edge_idx] = new_color
        new_c6s = count_c6s_from_adj(adj)
        
        if new_c6s < best_c6s:
            best_c6s = new_c6s
            best_colors = colors[:]
            best_adj = [row[:] for row in adj]
            no_improve_count = 0
        elif new_c6s == best_c6s and random.random() < 0.3:
            # Accept equal with some probability for diversity
            best_colors = colors[:]
            best_adj = [row[:] for row in adj]
            no_improve_count = 0
        else:
            # Revert
            colors[edge_idx] = old_color
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            no_improve_count += 1
        
        # If stuck for too long, try a random restart from best
        if no_improve_count > 5000:
            colors = best_colors[:]
            adj = [row[:] for row in best_adj]
            no_improve_count = 0
    
    if best_c6s == 0:
        return best_colors
    return None
>>>>>>> REPLACE

