"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    edges = []
    for u in part1:
        for v in part2:
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
