"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 15
    # Partition vertices into two sets A and B
    # A: 0..6 (7 vertices)
    # B: 7..14 (8 vertices)
    # Complete bipartite graph K_{7,8}
    # This ensures any 3-edge path s-u-v-t has s and t in different sets.
    # Degrees in A are 8, degrees in B are 7.
    # Since 8 != 7, endpoints of any 3-path have different degrees.
    
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
