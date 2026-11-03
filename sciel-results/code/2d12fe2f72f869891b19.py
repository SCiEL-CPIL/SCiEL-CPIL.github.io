def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Verify the full decomposition. U, V, W are numpy arrays."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = float(np.dot(u_row, v_row * w_row))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
