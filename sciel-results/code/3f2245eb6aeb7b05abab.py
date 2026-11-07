"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Return a 3-uniform hypergraph on 9 vertices with chromatic number exactly 3.
    
    Uses the Fano plane on vertices 0-6, with vertices 7 and 8 isolated.
    The Fano plane is the unique 3-uniform hypergraph with 7 edges that is 3-chromatic.
    """
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit

    # Build Fano plane edges using the structure of Z_7
    # Lines are {i, i+1, i+3} mod 7
    edge_set = set()
    for i in range(7):
        edge = tuple(sorted([(i) % 7, (i + 1) % 7, (i + 3) % 7]))
        edge_set.add(edge)
    
    edges = [list(e) for e in sorted(edge_set)]
    
    # Verify we have exactly 7 edges
    if len(edges) != 7:
        raise ValueError("Expected 7 edges, got " + str(len(edges)))
        
    # Assign a valid 3-coloring for vertices 0-6
    # Colors for 7, 8 can be anything, let's set them to 0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify the coloring is valid (each edge is nonmonochromatic)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if len({c0, c1, c2}) < 2:
             raise ValueError(f"Edge {e} is monochromatic")
             
    # Optional: Verify 3-chromaticity (not 2-colorable)
    # Precompute bit masks for faster checking
    edge_masks = []
    for e in edges:
        m = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
        edge_masks.append(m)
        
    is_2_colorable = False
    for mask in range(128):
        valid = True
        for m in edge_masks:
            # Check if all three bits in m are 0 or all are 1 in mask
            # If (mask & m) == 0 or (mask & m) == m, then it's monochromatic
            if (mask & m) == 0 or (mask & m) == m:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        raise ValueError("Hypergraph is 2-colorable, not 3-chromatic")

    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")

    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
