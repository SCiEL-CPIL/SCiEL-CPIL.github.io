        col_i = all_vectors[:, i]
        col_j = all_vectors[:, j]
        mask = col_i > col_j
        if np.any(mask):
            all_vectors[mask, i] = col_j[mask]
            all_vectors[mask, j] = col_i[mask]
