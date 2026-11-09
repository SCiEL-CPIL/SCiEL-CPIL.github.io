"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # Construct the Clebsch graph: a strongly regular graph srg(16, 5, 0, 2)
    # It is triangle-free (lambda=0) and has diameter 2 (mu=2).
    # Vertices are 4-bit vectors. Two vertices are adjacent iff their Hamming
    # distance is 1 or 4.
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            # Compute Hamming distance between u and v
            xor = u ^ v
            # Count bits set in xor
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
