def construct():
    def build_q3_free(n):
        if n <= 0:
            return [], 0
        if n == 1:
            return [], 0
        if n == 2:
            return [[0, 1]], 1
        if n == 3:
            return [[0, 1], [1, 2], [0, 2]], 3
        if n == 4:
            return [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], 6
        
        # Partition into A (size 3) and B (size n-3)
        size_a = 3
        size_b = n - 3
        
        edges_a = build_q3_free(size_a)[0]
        edges_b, _ = build_q3_free(size_b)
        
        # Shift edges in B by size_a
        edges_b_shifted = [[u + size_a, v + size_a] for u, v in edges_b]
        
        # Edges within A (already relative to 0..2)
        # Edges between A and B
        edges_ab = [[a, b + size_a] for a in range(size_a) for b in range(size_b)]
        
        all_edges = edges_a + edges_b_shifted + edges_ab
        return all_edges, len(all_edges)

    edges, _ = build_q3_free(16)
    return edges
