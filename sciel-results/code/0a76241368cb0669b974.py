"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 9, 'r': 3, 'colors': 3}

def construct():
    # Fano plane on vertices 0-6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    # Valid 3-coloring for Fano plane
    # v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1
    # v7, v8 isolated, color 0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
