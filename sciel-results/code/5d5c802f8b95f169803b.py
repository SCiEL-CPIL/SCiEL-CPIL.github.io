"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane: 7 edges on vertices 0-6, 3-uniform, chromatic number 3.
    # This is the theoretical minimum for any 3-uniform 3-chromatic hypergraph.
    # Vertices 7, 8 are isolated (any color).
    
    e = [[0, 1, 2],
         [0, 3, 4],
         [0, 5, 6],
         [1, 3, 5],
         [1, 4, 6],
         [2, 3, 6],
         [2, 4, 5]]
    
    c = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': e, 'colors': c}

# EVOLVE-BLOCK-END
