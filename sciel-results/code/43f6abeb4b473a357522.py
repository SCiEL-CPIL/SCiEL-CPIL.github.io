        # Precompute compatibility matrix
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edge_order)}
        comp_matrix = [[False] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if is_compatible(edge_order[i], edge_order[j]):
                    comp_matrix[i][j] = True
                    comp_matrix[j][i] = True
