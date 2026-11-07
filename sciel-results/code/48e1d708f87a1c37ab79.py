"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    
    # K_{8,8} is the unique extremal triangle-free graph on 16 vertices with 64 edges.
    # Any triangle would make an edge non-essential (third vertex provides length-2 path).
    # K_{8,8} has diameter exactly 2 and is the maximum-edge graph satisfying the constraints.
    A = list(range(half))
    B = list(range(half, n))
    
    edges = [[int(u), int(v)] for u in A for v in B]
    
    return edges
# EVOLVE-BLOCK-END
