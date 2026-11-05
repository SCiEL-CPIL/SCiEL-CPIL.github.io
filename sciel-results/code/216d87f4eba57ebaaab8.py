for i in range(n):
    for j in range(m):
        for k in range(p):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            if not np.isclose(val, 1.0, atol=1e-9):
                return False
