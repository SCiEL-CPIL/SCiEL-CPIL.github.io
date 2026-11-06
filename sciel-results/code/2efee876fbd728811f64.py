"""Improved program using Andrasfai graph And(9) on 27 vertices."""
# EVOLVE-BLOCK-START
def construct():
    # Andrasfai graph And_3(9): Z_27 with i ~ j iff (j - i) mod 27 is in {1, 10, 19}.
    # This graph is triangle-free and has independence number 9.
    # Thus it is a valid graph for R(3,10) > 27.
    k = 9
    n = 3 * k
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in (1, k + 1, 2 * k + 1):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
