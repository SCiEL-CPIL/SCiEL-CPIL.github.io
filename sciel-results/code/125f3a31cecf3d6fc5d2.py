"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices (0-8) that is 3-chromatic
    with the minimum number of edges, along with a valid 3-coloring.
    """
    
    # Candidate 1: 6-edge hypergraph (Theoretical minimum for 3-chromatic 3-uniform)
    # This specific configuration is known to be 3-chromatic.
    edges_6 = [
        [0, 1, 2],
        [0, 3, 4],
        [1, 3, 5],
        [2, 4, 5],
        [0, 5, 6],
        [1, 4, 6]
    ]
    
    colors_6 = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify 3-coloring validity (no monochromatic edges)
    def is_valid_3_coloring(edges, colors):
        for e in edges:
            c0 = colors[e[0]]
            c1 = colors[e[1]]
            c2 = colors[e[2]]
            if c0 == c1 == c2:
                return False
        return True

    # Verify 2-colorability (if it IS 2-colorable, it is NOT 3-chromatic)
    def is_2_colorable(edges, n_vertices):
        # Backtracking search for 2-coloring
        # We can fix color of vertex 0 to 0 due to symmetry
        colors = [None] * n_vertices
        colors[0] = 0
        
        # Adjacency list of constraints? 
        # Simpler: recursive check
        def backtrack(idx):
            if idx == n_vertices:
                # Check all edges
                for e in edges:
                    c0 = colors[e[0]]
                    c1 = colors[e[1]]
                    c2 = colors[e[2]]
                    if c0 == c1 == c2:
                        return False
                return True
            
            # Try 0
            colors[idx] = 0
            if backtrack(idx + 1):
                return True
            
            # Try 1
            colors[idx] = 1
            if backtrack(idx + 1):
                return True
            
            colors[idx] = None
            return False
        
        return backtrack(1)

    # Check Candidate 1
    if is_valid_3_coloring(edges_6, colors_6) and not is_2_colorable(edges_6, 9):
        return {
            'edges': edges_6,
            'colors': colors_6
        }
    else:
        # Fallback: Fano Plane (7 edges) on vertices 0-6, isolated 7-8
        # Fano plane is definitely 3-chromatic
        edges_7 = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 5, 6],
            [1, 3, 5],
            [1, 4, 6],
            [2, 3, 6],
            [2, 4, 5]
        ]
        
        # Valid 3-coloring for Fano plane
        # 0:0, 1:1, 2:2, 3:0, 4:1, 5:0, 6:2
        colors_7 = [0, 1, 2, 0, 1, 0, 2, 0, 0]
        
        return {
            'edges': edges_7,
            'colors': colors_7
        }

# EVOLVE-BLOCK-END
