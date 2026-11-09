# Inside _backtrack_optimized
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n_edges:
            return colors[:]
            
        i, j = edges[idx]
        
        # Determine allowed colors based on symmetry breaking
        if i == 0:
            # Vertex 0 is the "root"
            # We can fix the color of edge (0, j) to be < j
            # This assumes we process edges (0,1), (0,2), ... (0, n-1) first?
            # Our edge order is lexicographic: (0,1), (0,2), ..., (0, n-1), (1,2), ...
            # So for i=0, j is 1, 2, ..., n-1.
            # We can restrict c to be in {0, ..., j-1}
            # BUT we must ensure consistency with previous choices.
            # Actually, just restricting to c < j is a valid symmetry break for the first vertex.
            max_color = j
            candidates = list(range(max_color))
        else:
            candidates = list(range(5))
            random.shuffle(candidates)
            
        for c in candidates:
            if is_valid(i, j, c):
                colors[idx] = c
                add_edge(i, j, c)
                res = backtrack(idx + 1)
                if res:
                    return res
                remove_edge(i, j, c)
                colors[idx] = -1
                
        return None
