"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the Clebsch graph on 16 vertices.
    # Vertices are integers 0..15 (4-bit binary).
    # Edges exist if Hamming distance is 1 (hypercube Q4) or 4 (antipodal matching).
    edges = []
    n = 16
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            # Count bits
            bits = bin(xor).count('1')
            if bits == 1 or bits == 4:
                edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
