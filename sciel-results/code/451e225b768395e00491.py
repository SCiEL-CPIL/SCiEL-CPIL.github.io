"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Disjoint union of 8 copies of K4: C5-free, alpha = 8 < 9, n = 32.
    n = 32
    edges = []
    for block in range(8):
        base = 4 * block
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
