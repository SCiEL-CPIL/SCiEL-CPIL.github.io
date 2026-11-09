# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """Construct K_{8,8} complete bipartite graph on vertices 0..15."""
    n = P['n']
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
