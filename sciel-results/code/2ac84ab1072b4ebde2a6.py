import time
import random

def construct():
    """
    Returns a C5-free graph with no independent set of size 9.
    Best known construction: 8 disjoint copies of K4.
    n = 32.
    alpha = 8.
    C5-free: Yes.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
