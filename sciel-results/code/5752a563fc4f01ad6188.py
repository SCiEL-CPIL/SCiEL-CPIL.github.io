"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Andrásfai graph A_3(27)
    # n = 27
    # Edges: (i, j) if (j - i) % 27 in {1, 4, ..., 25}
    # This graph is triangle-free.
    # Its independence number is 9.
    # Thus, it has no independent set of 10 vertices.
    n = 27
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
