"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 vertices and 7 edges.
    # We map the Fano plane vertices to 0..6 and leave 7, 8 isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6:
    # c[0]=0, c[1]=1, c[2]=2
    # c[3]=1, c[4]=2, c[5]=0
    # c[6]=1
    # Vertices 7 and 8 are isolated, so they can be any color. We choose 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
