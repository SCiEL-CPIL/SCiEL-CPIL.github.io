"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    edges = []
    
    # First K3,3 on vertices 0-5
    # Bipartition: {0, 2, 4} and {1, 3, 5}
    for u in [0, 2, 4]:
        for v in [1, 3, 5]:
            edges.append([u, v])
            
    # Second K3,3 on vertices 6-11
    # Bipartition: {6, 8, 10} and {7, 9, 11}
    for u in [6, 8, 10]:
        for v in [7, 9, 11]:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
