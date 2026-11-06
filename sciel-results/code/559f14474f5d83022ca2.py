import random
import time

# EVOLVE-BLOCK-START
def construct():
    # The best known construction is 8 disjoint copies of K4.
    # This gives n=32, alpha=8 (no independent 9-set), and no C5 (each component is K4,
    # which has no 5-cycle; disjoint union preserves C5-freeness).
    #
    # We verify this is valid:
    # - No C5: each component is K4 (4 vertices, 6 edges). A C5 requires 5 distinct vertices
    #   in a cycle. Since each component has only 4 vertices, no component can contain a C5.
    #   Disjoint union of C5-free graphs is C5-free.
    # - No independent 9-set: the independence number of a disjoint union is the sum of
    #   independence numbers of components. alpha(K4) = 1, so alpha = 8 * 1 = 8 < 9.
    #
    # This matches the known best construction for R(C5, K9) > 32.
    #
    # We could try to search for larger graphs, but the task states the best known has 32
    # vertices and no valid graph has more than 35. Given the tight constraint and the
    # 150s time limit, we return the known optimal construction directly.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
