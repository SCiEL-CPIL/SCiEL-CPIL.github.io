"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    mid = n // 2
    edges = []
    # Construct complete bipartite graph K_{mid, n-mid}
    # For n=16, mid=8, so K_{8,8} with 64 edges.
    # Diameter is 2.
    # Removing any edge (u,v) makes u and v distance 3, so diameter becomes 3.
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
