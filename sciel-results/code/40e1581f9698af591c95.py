"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    edges = []
    # Component 1: K7 on vertices 0-6
    for i in range(7):
        for j in range(i + 1, 7):
            edges.append([i, j])
    # Component 2: K7 on vertices 7-13
    for i in range(7, 14):
        for j in range(i + 1, 14):
            edges.append([i, j])
    # Component 3: K2 on vertices 14-15
    edges.append([14, 15])
    
    return edges
# EVOLVE-BLOCK-END
