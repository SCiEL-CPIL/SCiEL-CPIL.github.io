"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices: 8 disjoint copies of K4.
    - No 5-cycles: K4 is C5-free, and disjoint union of C5-free graphs is C5-free.
    - No independent set of 9: alpha(K4)=1, so alpha(G) = 8 * 1 = 8 < 9.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
