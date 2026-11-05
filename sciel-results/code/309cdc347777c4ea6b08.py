import time
import random

def construct():
    """
    Constructs a graph with n vertices, no 5-cycles, and no independent set of size 9.
    Uses 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    Independence number is 8 (one vertex from each K4).
    No 5-cycles because K4 has no cycles of length 5 and components are disjoint.
    """
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
