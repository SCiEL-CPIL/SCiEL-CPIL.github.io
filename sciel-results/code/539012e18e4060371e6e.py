"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit

    # The Fano plane is the unique 3-chromatic hypergraph with 7 vertices and 7 edges.
    # We use vertices 0..6 for the Fano plane and leave 7..8 isolated.
    # This ensures the chromatic number is 3 (inherited from Fano) and edge count is minimal (7).
    
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

    # Verify: every edge must be non-monochromatic in the provided 3-coloring
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This should never happen for the Fano plane with this coloring
            # Fallback: return a known valid configuration if logic error occurs
            # But Fano plane is robust; this is just a safety net
            edges = [
                [0, 1, 2],
                [0, 3, 4],
                [0, 5, 6],
                [1, 3, 5],
                [1, 4, 6],
                [2, 3, 6],
                [2, 4, 5]
            ]
            colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
            break

    # Verify 3-chromaticity: exhaustively check that no 2-coloring exists
    # For 9 vertices, there are 2^9 = 512 possible 2-colorings.
    # A valid 2-coloring requires every edge to be non-monochromatic.
    # The Fano plane is known to be non-2-colorable, but we verify to be safe.
    found_2coloring = False
    for assignment in product([0, 1], repeat=9):
        valid = True
        for e in edges:
            c0, c1, c2 = assignment[e[0]], assignment[e[1]], assignment[e[2]]
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            found_2coloring = True
            break
    
    if found_2coloring:
        # This should never happen for the Fano plane, but if it does,
        # we need a different hypergraph. However, the Fano plane is
        # provably not 2-colorable, so this branch is unreachable.
        # As a safety measure, we still return the Fano plane since it's
        # the best we have and the evaluator will reject it if truly 2-colorable.
        pass

    # Ensure types are correct: integers must be Python int, sequences must be list or tuple
    edges = [list(map(int, e)) for e in edges]
    colors = list(map(int, colors))

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
