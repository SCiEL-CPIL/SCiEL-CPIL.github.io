"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # n = 8 * 4 = 32.
    # alpha(G) = 8, which is < 9.
    # K4 has no C5, so the union has no C5.
    blocks = 8
    size = 4
    n = blocks * size
    edges = []
    for b in range(blocks):
        base = b * size
        for i in range(size):
            for j in range(i + 1, size):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
