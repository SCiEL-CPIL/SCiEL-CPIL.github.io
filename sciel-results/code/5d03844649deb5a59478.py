"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    added_edges = []
    
    # Strategy: Connect vertex 1 (in B) to all even vertices (A) except those already connected.
    # This creates a bipartite graph with diameter 4.
    # Vertices: 0..23.
    # A = {0, 2, ..., 22}, B = {1, 3, ..., 23}.
    # Center at 1 (B). Connect to all A.
    # Path edges from 1 to A are (1,0) and (1,2).
    # So add (1, a) for a in A if (1, a) not in base.
    
    for a in range(0, n, 2):
        if (1, a) not in base_edges:
            added_edges.append([1, a])
            
    return added_edges
# EVOLVE-BLOCK-END
