"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the minimal non-2-colorable 3-uniform hypergraph.
    # It has 7 vertices and 7 edges.
    # We use vertices 0-6 for the Fano plane and leave 7-8 isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane part:
    # c(0)=0, c(1)=1, c(2)=2
    # c(3)=0, c(4)=1, c(5)=2
    # c(6)=0
    # Isolated vertices 7, 8 can be any color.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Convert to lists and ensure Python int types
    edges_list = [list(e) for e in edges]
    colors_list = [int(c) for c in colors]
    
    return {'edges': edges_list, 'colors': colors_list}

# EVOLVE-BLOCK-END
