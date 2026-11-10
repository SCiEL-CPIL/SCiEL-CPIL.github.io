import time

# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            # Skip one edge to ensure diameter is exactly 2 (not 1)
            # Removing any single edge from K_n (n>=3) leaves diameter 2
            # and all remaining edges are in triangles.
            if u == 0 and v == 1:
                continue
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
