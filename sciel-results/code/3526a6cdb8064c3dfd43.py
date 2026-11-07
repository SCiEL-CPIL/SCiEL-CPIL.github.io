"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-uniform hypergraph on vertices 0..8 with the minimum number of edges
    such that the chromatic number is exactly 3.
    
    Theoretical minimum is 7 edges (Fano plane).
    """
    # Fano plane edges on vertices 0..6
    # Vertices 7 and 8 are isolated
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring for the Fano plane
    # 0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 1
    # 7: 0, 8: 0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify that no edge is monochromatic in this coloring
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            raise ValueError("Invalid coloring: monochromatic edge found")
            
    # Verify that the hypergraph is NOT 2-colorable (i.e., chromatic number > 2)
    # This is a property of the Fano plane, but we can double-check with a small search
    # for 2-colorings of the 7 vertices involved.
    # If a 2-coloring exists, it would assign 0 or 1 to each vertex in 0..6.
    # There are 2^7 = 128 possible colorings.
    is_2_colorable = False
    for c in itertools.product([0, 1], repeat=7):
        valid = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        raise ValueError("Hypergraph is 2-colorable, chromatic number is not 3")

    # Convert to required types: list of lists for edges, list of ints for colors
    edges_out = [list(e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
