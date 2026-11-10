"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""

def construct():
    """
    Return a 5-uniform hypergraph on n vertices with no proper 2-colouring,
    minimizing the number of edges.
    
    Uses a greedy edge removal strategy starting from the complete 5-uniform 
    hypergraph on 9 vertices (which is known to be non-2-colorable).
    """
    import itertools
    
    def is_not_2_colorable(n, edges):
        """
        Check if hypergraph is NOT 2-colorable.
        For small n (<= 10), use brute force over all 2^n colorings.
        A hypergraph is 2-colorable if there exists a coloring where 
        no edge is monochromatic.
        """
        m = len(edges)
        if m == 0:
            return False
        
        # Precompute edges as tuples for faster access
        edge_list = [tuple(e) for e in edges]
        
        # Iterate through all 2^n colorings
        for mask in range(1 << n):
            # Check if this coloring works
            # Coloring v is (mask >> v) & 1
            is_valid = True
            for e in edge_list:
                # Check if edge is monochromatic
                # All vertices in e must have the same bit in mask
                first_color = (mask >> e[0]) & 1
                mono = True
                for v in e[1:]:
                    if ((mask >> v) & 1) != first_color:
                        mono = False
                        break
                if mono:
                    is_valid = False
                    break
            
            if is_valid:
                # Found a valid 2-coloring, so it IS 2-colorable
                return False
        
        # No valid coloring found, so it is NOT 2-colorable
        return True

    # Start with complete 5-uniform hypergraph on 9 vertices
    n_start = 9
    all_edges = list(itertools.combinations(range(n_start), 5))
    
    # Greedy removal
    current_edges = all_edges[:]
    
    # To make it more effective, we can try multiple passes
    for _ in range(10):
        i = 0
        changed = False
        while i < len(current_edges):
            # Try removing current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if still non-2-colorable
            if is_not_2_colorable(n_start, new_edges):
                # Removal successful
                current_edges = new_edges
                changed = True
                # Do not increment i, check the new edge at position i
            else:
                # Removal failed, keep edge
                i += 1
        if not changed:
            break

    return {'n': n_start, 'edges': [list(e) for e in current_edges]}
