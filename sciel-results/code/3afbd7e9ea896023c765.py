"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges.
    # It is known to be 3-chromatic (cannot be colored with 2 colors, but can with 3).
    # We embed it on vertices 0..6 and leave vertices 7 and 8 isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane:
    # c[0]=0, c[1]=1, c[2]=2, c[3]=0, c[4]=1, c[5]=0, c[6]=2
    # Vertices 7 and 8 can be any color, we choose 0.
    colors = [0, 1, 2, 0, 1, 0, 2, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
