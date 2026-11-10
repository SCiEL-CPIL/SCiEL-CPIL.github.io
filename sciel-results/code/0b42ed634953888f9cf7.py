"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    n = P['n']
    edges = []
    for i in range(n):
        for j in range(i):
            edges.append([i, j])
    return edges
# EVOLVE-BLOCK-END
