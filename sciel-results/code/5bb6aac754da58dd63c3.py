"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Fano plane on vertices 0-6: 7 edges, known to be 3-chromatic
    # Vertices 7,8 are isolated
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring: no edge is monochromatic
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify: every edge must be non-monochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # Fallback to seed if somehow invalid (should not happen)
            from itertools import combinations
            return {'edges': [list(e) for e in combinations(range(5), 3)],
                    'colors': [0, 0, 1, 1, 2, 0, 0, 0, 0]}
    
    # Verify types
    edges = [list(e) for e in edges]
    colors = list(colors)
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
