"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # Partition vertices 0..15 into two sets of size 8
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
