"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return a minimal 3-uniform 3-chromatic hypergraph on vertices 0..8.
    Uses the Fano plane (7 edges) which is provably optimal for 3-uniform
    3-chromatic hypergraphs. Vertices 7,8 are isolated.
    """
    # Fano plane edges via cyclic construction mod 7
    # Base block: {0, 1, 3}; cyclic shifts give all 7 lines
    blocks = []
    for shift in range(7):
        triple = tuple(sorted((shift + 0) % 7, (shift + 1) % 7, (shift + 3) % 7))
        blocks.append(triple)
    
    # Deduplicate (should already be unique, but safe)
    seen = set()
    edges = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            edges.append([int(b[0]), int(b[1]), int(b[2])])
    
    # 3-coloring: assign colors so no edge is monochromatic
    # c: 0->0, 1->1, 2->2, 3->1, 4->2, 5->0, 6->1, 7->0, 8->0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
