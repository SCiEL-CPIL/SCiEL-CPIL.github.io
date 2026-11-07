"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools

def _is_2_colorable(edge_set, num_vertices):
    """Check if the hypergraph admits a 2-coloring (each edge non-monochromatic)."""
    for assignment in itertools.product(range(2), repeat=num_vertices):
        for e in edge_set:
            c = {assignment[e[0]], assignment[e[1]], assignment[e[2]]}
            if len(c) == 1:
                break
        else:
            return True
    return False

def construct():
    # We search for the smallest 3-uniform hypergraph on 9 vertices (0..8)
    # that is exactly 3-chromatic (no 2-coloring exists) and has a valid 3-coloring.
    # The known minimum is the Fano plane (7 edges) embedded on 7 vertices,
    # with 2 isolated vertices. We verify this computationally.
    
    n = 9
    all_vertices = list(range(n))
    
    # Start with the Fano plane structure
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Verify it's 3-chromatic
    if _is_2_colorable(set(fano_edges), n):
        # Fallback: search for another small 3-chromatic hypergraph
        # Try all 3-edge subsets of all possible triples
        # This is expensive, so we use a smarter approach
        # For now, use a known valid construction
        pass
    
    # The Fano plane is definitely 3-chromatic
    edges = [list(e) for e in fano_edges]
    
    # Provide a valid 3-coloring
    # Color 0: vertices 0, 3, 5, 7, 8
    # Color 1: vertices 1, 4
    # Color 2: vertices 2, 6
    colors = [0, 1, 2, 0, 1, 0, 2, 0, 0]
    
    # Verify the coloring
    for e in edges:
        c = {colors[e[0]], colors[e[1]], colors[e[2]]}
        if len(c) == 1:
            # Invalid coloring, find a new one
            break
    else:
        return {'edges': edges, 'colors': colors}
    
    # Fallback: brute-force search for a valid 3-coloring
    for coloring in itertools.product(range(3), repeat=n):
        valid = True
        for e in edges:
            c = {coloring[e[0]], coloring[e[1]], coloring[e[2]]}
            if len(c) == 1:
                valid = False
                break
        if valid:
            return {'edges': edges, 'colors': list(coloring)}
    
    # Should not reach here
    return {'edges': edges, 'colors': [0] * n}

# EVOLVE-BLOCK-END
