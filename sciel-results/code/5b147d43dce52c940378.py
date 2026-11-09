# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum number of edges (7).
    
    The Fano plane is the unique 3-uniform 3-chromatic hypergraph with 7 edges.
    We construct it using the cyclic group Z_7 structure.
    """
    # Generate Fano plane edges on vertices 0-6
    edges = []
    for i in range(7):
        # Each edge is {i, i+1, i+3} mod 7
        e = [i, (i + 1) % 7, (i + 3) % 7]
        # Sort for consistency, though not strictly required if distinctness is checked
        e.sort()
        edges.append(e)
        
    # Define a valid 3-coloring for the Fano plane vertices 0-6
    # Colors: 0, 0, 1, 1, 2, 2, 0
    colors_fano = [0, 0, 1, 1, 2, 2, 0]
    
    # Vertices 7 and 8 are isolated, assign color 0
    colors = colors_fano + [0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
