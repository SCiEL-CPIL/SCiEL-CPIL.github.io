def construct():
    N = 15
    # Deterministic construction: Complete Bipartite Graph K_{7,8}
    # Vertices 0..6 in set A, 7..14 in set B.
    # All edges between A and B are included.
    # No edges within A or within B.
    # Degrees in A are 8.
    # Degrees in B are 7.
    # For any path s-u-v-t, endpoints s and t are in different sets (A and B).
    # Thus d(s) != d(t) is always true (8 != 7).
    # This graph is valid.
    # Edge count: 7 * 8 = 56.
    
    edges = []
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
            
    return edges
