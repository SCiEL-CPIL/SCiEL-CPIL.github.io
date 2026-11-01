def compute_residual(U, V, W, rank):
    R = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    R -= 1.0
    return R
