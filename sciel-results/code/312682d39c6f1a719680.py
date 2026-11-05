"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    edges = []
    # Bipartite graph with parts A={0, 1} and B={2, ..., 19}
    # We remove edges (0, v) for v in {12, ..., 19} to break bad 4-cycles.
    # Bad 4-cycles correspond to pairs (v_a, v_b) in B such that v_b - v_a = 10.
    # Specifically, indices j_a, j_b with j_b = j_a + 10.
    # j in 0..17. Pairs: (0,10), (1,11), ..., (7,17).
    # Vertices in B: 2..19. Index j = v - 2.
    # j=0 -> v=2. j=10 -> v=12.
    # So pairs of vertices: (2,12), (3,13), ..., (9,19).
    # We remove edge (0, v_b) for v_b in {12, 13, 14, 15, 16, 17, 18, 19}.
    
    removed_from_0 = set(range(12, 20))
    
    for v in range(2, 20):
        j = v - 2
        # Edge (0, v)
        if v not in removed_from_0:
            edges.append([0, v, j])
        # Edge (1, v)
        edges.append([1, v, (j + 10) % 20])
        
    return edges
# EVOLVE-BLOCK-END
