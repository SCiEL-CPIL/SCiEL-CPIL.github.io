def construct():
    """
    Constructs a C5-free graph with independence number < 9.
    
    Strategy:
    Use the disjoint union of 8 copies of K4.
    - Vertices: 8 * 4 = 32.
    - C5-free: K4 has no 5-cycle. Disjoint union preserves C5-freeness.
    - Independence number: alpha(K4) = 1. Total alpha = 8 * 1 = 8.
    - 8 < 9, so no independent set of size 9 exists.
    
    This achieves n=32, which is the best known lower bound for R(C5, K9) > n.
    """
    n = 32
    edges = []
    
    # Create 8 disjoint cliques of size 4
    for b in range(8):
        base = b * 4
        # Generate all edges within the clique {base, base+1, base+2, base+3}
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
