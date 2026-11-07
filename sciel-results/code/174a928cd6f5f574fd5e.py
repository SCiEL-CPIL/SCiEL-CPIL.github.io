"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    edges = []
    # Complement is a perfect matching: (0,1), (2,3), ..., (14,15)
    # We generate all edges in K_16 except these 8.
    
    # Create a set of forbidden edges (the matching)
    forbidden = set()
    for i in range(0, n, 2):
        forbidden.add((i, i + 1))
        forbidden.add((i + 1, i))

    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in forbidden:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
