"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _check_2_colorable(edges):
    """Check if the hypergraph is 2-colorable. Returns True if it is."""
    n_vertices = 9
    # Represent coloring as a bitmask or tuple. 2^9 = 512 possibilities.
    # For each coloring, check if all edges are non-monochromatic.
    for mask in range(1 << n_vertices):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            # If all three are the same, it's monochromatic
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    # Fano plane edges on vertices 0..6
    # Vertices 7, 8 are isolated
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verified 3-coloring for the Fano plane
    # c[0]=0, c[1]=1, c[2]=2
    # c[3]=1, c[4]=2, c[5]=0
    # c[6]=1
    # c[7]=0, c[8]=0 (isolated)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Sanity Check 1: Ensure provided coloring is valid (no monochromatic edges)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Provided coloring is invalid: monochromatic edge found")
            
    # Sanity Check 2: Ensure it is NOT 2-colorable
    if _check_2_colorable(edges):
        raise ValueError("Graph is 2-colorable, violates 3-chromatic requirement")

    # Ensure types are strictly Python int and list
    final_edges = [tuple(e) for e in edges]
    final_colors = [int(c) for c in colors]
    
    return {'edges': final_edges, 'colors': final_colors}

# EVOLVE-BLOCK-END
