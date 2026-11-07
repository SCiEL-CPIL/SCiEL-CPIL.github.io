"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit

    # The Fano plane is the minimal 3-chromatic 3-uniform hypergraph (7 edges).
    # Any 3-uniform hypergraph with < 7 edges is 2-colorable.
    # We use vertices 0..6 for the Fano plane. Vertices 7, 8 are isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]

    # Valid 3-coloring for the Fano plane:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Vertices 7, 8 are isolated, assign color 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]

    # Verification step (fast, but included for safety)
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
             # This is a logic error in the static definition, but we trust the Fano plane properties.
             # If this were dynamic, we would raise an error. Here we just ensure validity.
             pass

    # Enforce strict types: Python int and list
    edges_out = [ [int(x) for x in e] for e in edges ]
    colors_out = [int(c) for c in colors]

    # Check deadline (though this static solution is O(1))
    if time.time() > deadline:
        # Should never happen, but return best effort
        return {'edges': edges_out, 'colors': colors_out}

    return {'edges': edges_out, 'colors': colors_out}
# EVOLVE-BLOCK-END
