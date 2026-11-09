"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 9, 'r': 3, 'colors': 3}

def construct():
    # Fano plane on vertices 0-6, embedded in 9 vertices (7,8 isolated)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    # 3-coloring: 0->0, 1->1, 2->2, 3->0, 4->1, 5->2, 6->0, 7->0, 8->0
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
