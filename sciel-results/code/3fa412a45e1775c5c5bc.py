"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # K_{8,8} is the unique triangle-free graph on 16 vertices with maximum edges (64).
    # By Turán's theorem, max edges in triangle-free graph on n vertices is floor(n^2/4) = 64.
    # Triangle-free is necessary: if u,v,w form a triangle, removing edge (u,v) leaves
    # path u-w-v of length 2, so (u,v) is not critical.
    # In K_{8,8}, removing any edge (u,v) with u in part A, v in part B makes
    # dist(u,v) = 3 (via any other vertex), so diameter becomes 3 > 2. Every edge is critical.
    # 64 is the theoretical maximum; this is optimal.
    
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([a, b])
    
    return edges
# EVOLVE-BLOCK-END
