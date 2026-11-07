"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges
    # that is not 2-colorable (chromatic number 3).
    # We embed it in 9 vertices (vertices 7 and 8 are isolated).
    # This is the minimum number of edges for a 3-chromatic 3-uniform hypergraph.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring of the Fano plane
    # Colors are 0, 1, 2
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are exactly Python int and list
    edges_out = [list(e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    # Verification / Polish Pass
    # Check if we are still within time limit
    if time.time() > deadline:
        # If somehow we are close to timeout (unlikely for static code), return immediately
        pass
        
    # Verify constraints to ensure validity
    # 1. Distinct triples
    assert len(set(tuple(e) for e in edges_out)) == len(edges_out)
    # 2. Vertices in 0..8
    for e in edges_out:
        assert all(0 <= v <= 8 for v in e)
    # 3. Colors in 0..2
    assert all(0 <= c <= 2 for c in colors_out)
    # 4. Non-monochromatic edges
    for e in edges_out:
        c0, c1, c2 = colors_out[e[0]], colors_out[e[1]], colors_out[e[2]]
        assert not (c0 == c1 == c2)
        
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
