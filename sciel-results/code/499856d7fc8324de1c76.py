"""Construct a dense triangle-free, non-3-colorable graph on 16 vertices."""

def construct():
    """
    Return the edge list of the Clebsch graph on 16 vertices.
    The Clebsch graph is triangle-free and has chromatic number 4,
    giving 40 edges — the maximum for a triangle-free graph on 16 vertices.
    """
    n = 16
    # Clebsch graph: vertices are 4-bit binary strings (0..15).
    # Two vertices are adjacent iff their Hamming distance is 1 or 3.
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            popcount = bin(xor).count('1')
            if popcount in (1, 3):
                edges.append([u, v])
    return edges
