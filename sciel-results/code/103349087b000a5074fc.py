"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # Partition vertices into two sets of 8
    U = list(range(0, 8))
    V = list(range(8, 16))
    
    edges = []
    for u in U:
        for v in V:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
