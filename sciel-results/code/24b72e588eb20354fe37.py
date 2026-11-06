def max_clique_size(adj_mask, n, limit):
    """
    Returns True if there is a clique of size >= limit.
    adj_mask[i] is the bitmask of neighbors of i in the graph.
    """
    # We use a recursive branch and bound.
    # State: current_clique_size, candidates_mask
    
    # Precompute greedy coloring for initial set?
    # We can compute it on the fly for the candidates.
    
    def greedy_color(candidates_mask):
        # Returns an upper bound on clique size in candidates_mask
        # Simple greedy coloring of the subgraph induced by candidates_mask
        # The number of colors used is an upper bound on the clique size?
        # No, clique size <= number of colors in a proper coloring.
        # So we want to minimize colors to get a tight bound.
        
        colors = {}
        uncolored = candidates_mask
        num_colors = 0
        
        while uncolored:
            num_colors += 1
            # Greedily pick a set of non-adjacent vertices (independent set in original, clique in comp?)
            # Wait, we are finding a CLIQUE in the graph defined by adj_mask.
            # So we need to color the graph such that adjacent vertices have different colors.
            # The clique number is at most the chromatic number.
            
            # To get a good bound, we want a small chromatic number.
            # Greedy coloring:
            # Pick a vertex, assign color num_colors, remove it and its neighbors?
            # No, that's for independent set.
            # For coloring:
            # We assign colors such that no two adjacent vertices have the same color.
            
            # Let's use a simple heuristic:
            # Order vertices by degree in the subgraph.
            # Assign colors greedily.
            
            # For speed, we can just use the size of the candidates as a loose bound
            # if the search is fast enough.
            # Given n=43, let's try the simple bound first.
            # If it's too slow, we add coloring.
            
            # Actually, for dense graphs, the simple bound (popcount) is often sufficient
            # because the branching factor is low?
            # No, branching factor is high.
            
            # Let's implement a simple greedy coloring bound.
            # We want to find a clique.
            # The bound is: max clique size in candidates <= chromatic number of candidates.
            
            # Greedy coloring algorithm:
            # 1. Order vertices in candidates by descending degree in the subgraph.
            # 2. For each vertex, assign the smallest possible color.
            
            # This is O(n^2).
            
            verts = [i for i in range(n) if (candidates_mask >> i) & 1]
            # Sort by degree in subgraph
            def deg_in_sub(v):
                return bin(adj_mask[v] & candidates_mask).count('1')
            
            verts.sort(key=deg_in_sub, reverse=True)
            
            color_of = {}
            next_color = 0
            for v in verts:
                # Find available colors
                used_colors = set()
                for u in verts:
                    if u == v: continue
                    if (adj_mask[v] >> u) & 1: # u is neighbor of v
                        if u in color_of:
                            used_colors.add(color_of[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                color_of[v] = c
                next_color = max(next_color, c + 1)
                
            return next_color

    def search(clique_size, candidates_mask):
        if clique_size >= limit:
            return True
            
        if candidates_mask == 0:
            return False
            
        # Bound
        # Simple bound
        if clique_size + bin(candidates_mask).count('1') < limit:
            return False
            
        # Greedy coloring bound
        # This might be expensive to compute every time.
        # We can compute it only if the simple bound is not pruning.
        # Or we can use a precomputed bound?
        # Let's try to use the simple bound first and see if it's fast enough.
        # If not, we can optimize.
        
        # To speed up, we can pick the vertex with the highest degree in the subgraph.
        # Or just the first one.
        
        # Pick a vertex
        # Let's pick the vertex with the highest degree in the candidate subgraph to prune faster.
        best_v = -1
        best_deg = -1
        cm = candidates_mask
        while cm:
            v = (cm & -cm).bit_length() - 1
            d = bin(adj_mask[v] & candidates_mask).count('1')
            if d > best_deg:
                best_deg = d
                best_v = v
            cm &= cm - 1
            
        if best_v == -1:
            return False
            
        # Branch 1: Include best_v
        new_candidates = candidates_mask & adj_mask[best_v]
        if search(clique_size + 1, new_candidates):
            return True
            
        # Branch 2: Exclude best_v
        candidates_mask &= ~(1 << best_v)
        if search(clique_size, candidates_mask):
            return True
            
        return False

    initial_mask = (1 << n) - 1
    return search(0, initial_mask)
