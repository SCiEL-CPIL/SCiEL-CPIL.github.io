"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """Construct the Clebsch graph (srg(16,5,0,2)) on vertices 0..15.

    The Clebsch graph is the folded 5-cube: vertices are 4-bit strings, and
    two vertices are adjacent if they differ in exactly 1 or exactly 3 bits.
    It is strongly regular with parameters (16,5,0,2), triangle-free, and
    every edge is critical for maintaining diameter 2.
    """
    n = 16
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            # Popcount of xor: adjacent if distance is 1 or 3 in 4-bit space
            if bin(xor).count('1') in (1, 3):
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
