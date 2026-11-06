import time

def construct():
    """
    Returns a C5-free graph with no independent set of size 9.
    Best known simple construction: 8 disjoint copies of K4.
    n = 32.
    alpha = 8.
    C5-free: Yes, as K4 does not contain C5.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
