"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Hub-and-spoke: connect vertex 12 to all odd vertices
    # This keeps the graph bipartite (triangle-free) and gives diameter 4.
    # dist(u, 12) <= 2 for all u, so dist(u, v) <= 4 for all u, v.
    added = []
    for v in range(1, n, 2):  # odd vertices: 1,3,5,...,23
        if (12, v) not in base_edges and (v, 12) not in base_edges:
            added.append((12, v))
    
    return [[int(u), int(v)] for u, v in added]

# EVOLVE-BLOCK-END
