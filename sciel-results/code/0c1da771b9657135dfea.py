        m = num_nodes
        r = np.sum(dist, axis=1)
        # Q[i,j] = (m-2)*d[i,j] - r[i] - r[j]
        # We can construct this matrix
        # r[:, None] - r[None, :] gives shape (m, m)
        # dist is (m, m)
        Q = (m - 2) * dist - (r[:, None] + r[None, :])
        # We only care about i < j
        # Set diagonal and lower triangle to inf
        np.fill_diagonal(Q, np.inf)
        Q[np.tril_indices(m)] = np.inf
        # Find min
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        best_i, best_j = min_idx
