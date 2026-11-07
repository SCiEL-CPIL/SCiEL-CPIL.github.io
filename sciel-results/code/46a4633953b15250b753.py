"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices with 7 edges (Fano plane + 2 isolated vertices)
    which is not 2-colorable, and a valid 3-coloring.
    """
    # Fano plane edges on vertices 0-6
    # This is the unique (up to isomorphism) 3-chromatic 3-uniform hypergraph with 7 edges.
    # It is the minimum possible edge count for this property.
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Colors for vertices 0-8
    # c(0)=0, c(1)=1, c(2)=2
    # c(3)=1, c(4)=1
    # c(5)=2, c(6)=2
    # c(7)=0, c(8)=0
    colors = [0, 1, 2, 1, 1, 2, 2, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
