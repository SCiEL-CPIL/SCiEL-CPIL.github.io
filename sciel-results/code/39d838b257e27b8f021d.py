"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We construct the complement of the disjoint union of 8 K4s.
    # H = 8 * K4.
    # G = complement(H).
    # H has 32 vertices.
    # H is C5-free. G is C5-free (C5 is self-complementary).
    # alpha(H) = 8.
    # alpha(G) = omega(H) = 4.
    # 4 < 9, so alpha(G) < 9.
    # n = 32.
    
    n = 32
    edges = []
    # Vertices are 0..31.
    # In H, edges exist within blocks of 4.
    # In G, edges exist between different blocks.
    for u in range(n):
        for v in range(u + 1, n):
            block_u = u // 4
            block_v = v // 4
            if block_u != block_v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
