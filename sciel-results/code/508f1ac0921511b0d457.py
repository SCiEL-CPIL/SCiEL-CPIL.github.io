"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 9, 'r': 3, 'colors': 3}

def construct():
    # Fano plane: 7 edges on 7 vertices (0..6), 3-chromatic, cannot be 2-colored
    # Valid 3-coloring:
    # v0=0, v1=1, v2=2, v3=2, v4=1, v5=1, v6=0
    # Edge checks:
    # (0,1,2): 0,1,2 OK
    # (0,3,4): 0,2,1 OK
    # (0,5,6): 0,1,0 OK
    # (1,3,5): 1,2,1 OK
    # (1,4,6): 1,1,0 OK
    # (2,3,6): 2,2,0 OK
    # (2,4,5): 2,1,1 OK
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    # 9 vertices: vertices 7, 8 are isolated (not in any edge), can be any color
    colors = [0, 1, 2, 2, 1, 1, 0, 0, 0]
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
