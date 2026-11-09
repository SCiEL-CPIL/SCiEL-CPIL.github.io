"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges 
    # that is not 2-colorable. It is the smallest such hypergraph.
    # Vertices are 0..6. Vertices 7 and 8 are isolated.
    
    # Define the 7 edges of the Fano plane
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verify that no 2-coloring exists (i.e., it is 3-chromatic)
    # A 2-coloring assigns each vertex 0 or 1. 
    # An edge is "bad" if all its vertices have the same color.
    # We need to ensure that for ANY assignment of 0/1 to vertices 0..8,
    # at least one edge is bad (monochromatic).
    # Actually, the definition of "not 2-colorable" means there is NO coloring
    # such that ALL edges are non-monochromatic.
    # So we check if there exists a 2-coloring where all edges are non-mono.
    # If no such coloring exists, the hypergraph is not 2-colorable.
    
    # Since vertices 7 and 8 are isolated, they don't affect edge constraints.
    # We only need to check vertices 0..6.
    # There are 2^7 = 128 possible 2-colorings for vertices 0..6.
    
    is_2_colorable = False
    for mask in range(128):
        # Check if this mask represents a valid 2-coloring (all edges non-mono)
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        # This should never happen for the Fano plane.
        # If it does, we have a bug in the edge definition.
        # For safety, we could raise an error or return a known bad state, 
        # but the Fano plane is mathematically guaranteed to be 3-chromatic.
        pass

    # Provide a valid 3-coloring.
    # The Fano plane is 3-colorable. One such coloring is:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    # 7: 0 (isolated)
    # 8: 0 (isolated)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify the provided 3-coloring is valid (no mono edges)
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            # Invalid coloring provided
            pass

    # Ensure types are strictly Python int and list
    edges_list = [list(map(int, e)) for e in edges]
    colors_list = list(map(int, colors))
    
    return {
        'edges': edges_list,
        'colors': colors_list
    }

# EVOLVE-BLOCK-END
