"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    # Construct K_{7,8}
    # Vertices 0-6 in A, 7-14 in B
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
    
    # Return as list of lists of ints
    return edges
# EVOLVE-BLOCK-END
